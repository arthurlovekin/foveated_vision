import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule
from torchvision.transforms import Resize
from torchinfo import summary
import math
import logging
# from typing import Tensor

RESNET_DEFAULT_INPUT_SIZE = (224, 224)

class PeripheralModel(nn.Module):
    """
    Creates a feature vector from a low-resolution background image
    using a pre-trained ResNet50 model. (These features are later combined
    with the foveal features to produce a bounding box and fixation point).
    """

    def __init__(self):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.pretrained, "fc"):
            self.pretrained.fc = torch.nn.Identity()
        else:
            logging.error("No fc layer found in pretrained model")
        # self.pretrained.fc = nn.Linear(2048, 4)
        # self.pretrined_new = torch.nn.Sequential(*list(self.pretrained.children())[:-1])
        # for param in self.pretrained[-1].parameters():
        #     param.requires_grad = False

    def forward(self, low_res_image):
        # output: (batch, 2048) feature vector
        return self.pretrained(low_res_image)


class FovealModel(nn.Module):
    """
    Creates a feature vector from a single high-resolution foveal patch.
    (These features are later combined
    with the peripheral features to produce a bounding box and fixation point).
    """

    def __init__(self):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.pretrained, "fc"):
            self.pretrained.fc = torch.nn.Identity()
        else:
            logging.error("No fc layer found in pretrained model")

    def forward(self, foveal_patch):
        # output: (batch, 2048) feature vector
        return self.pretrained(foveal_patch)

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/tatp22/multidim-positional-encoding for 2d and 3d positional encodings
class PositionalEncoding(nn.Module):
    """
        d_model (int) – the number of expected features in the encoder/decoder inputs
        dropout (float) – the dropout probability
        max_len (int) – the maximum length of the input feature
    """
    def __init__(self, d_model: int=512, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        position_encoding = torch.zeros(max_len, 1, d_model)
        position_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        """ x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.position_encoding[:x.size(0)]
        return self.dropout(x)

class CombinerModel(nn.Module):
    """
    Takes buffers of foveal features, peripheral features, and fovea points, and
    combines them to produce a bounding box and new fixation point.
    """

    def __init__(self, buffer_size=3, n_inputs: int=4098, n_heads: int=6,
                 n_encoder_layers: int=3, dropout: float = 0.1):
        """
        Args:
            buffer_size (int): Number of previous frames to consider
            n_inputs (int): Number of input features (peripheral features + foveal features + fovea points)
            n_heads (int): Number of attention heads. Must evenly divide n_inputs*batch_size
            n_encoder_layers (int): Number of encoder layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.positional_encoding = PositionalEncoding(n_inputs,dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=n_inputs, nhead=n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_model = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_encoder_layers) # (contains multiple TransformerEncoderLayers)
        # Map the encoded sequence to a bounding box.
        # TODO: This could use a more advanced decoder
        self.sequence_dim = buffer_size*n_inputs
        self.bbox_head = nn.Sequential(
            nn.Linear(self.sequence_dim, 4),
            nn.Sigmoid(), # make outputs 0-1
        )
        self.pos_head = nn.Sequential(
            nn.Linear(self.sequence_dim, 2),
            nn.Sigmoid(), # make outputs 0-1
        )

    def forward(self, all_features_buffer):
        # Transformer expects input of shape (batch, seq_len, feature_len)
        # Concatenate all features along the time dimension
        positional_features = self.positional_encoding(all_features_buffer)
        logging.debug(f"Transformer input shape: {positional_features.shape}")
        latent_seq = self.transformer_model(positional_features)
        # Hack decoder
        # Flatten the sequence
        latent_seq = latent_seq.reshape((latent_seq.shape[0], -1))
        bbox = self.bbox_head(latent_seq)
        pos = self.pos_head(latent_seq)
        return bbox, pos 

class CombinerModelTimeSeriesTransformer(nn.Module):
    """Uses a pre-trained time-series transformer (Autoformer) to combine the features from
    the peripheral and foveal models.
    """
    def __init__(self):
        super().__init__()
        self.transformer = None
        self.object_tracking_head = nn.Sequential(
            nn.Linear(2048, 4),
            nn.Sigmoid(),
        )

    def forward(self, all_features_buffer):
        """ Outputs both the predicted bounding box, 
        and the output attentions of the transformer, which are used by the NeuralFoavationModule to generate a foveal patch.
        """
        pass

class PeripheralFovealVisionModel(nn.Module):
    def __init__(self):#, batch_size=1):
        super().__init__()
        # self.batch_size = batch_size
        self.peripheral_resolution = RESNET_DEFAULT_INPUT_SIZE
        self.downsampler = Resize(
            (self.peripheral_resolution[0], self.peripheral_resolution[1]),
            antialias=True,
        )
        self.foveation_module = FoveationModule(bound_crops=False)
        self.peripheral_model = PeripheralModel()
        self.foveal_model = FovealModel()
        self.combiner_model = CombinerModel()
        self.position_encoding = None
        self.fixation_length = 2
        self.current_fixation = None 
        # self.feature_len = 2x2048 (resnet output) + 2 for center of fixation 
        self.buffer_len = 3
        self.buffer = None 
    
    def reset(self):
        # Need to detach then reattach the hidden variables of the model to prevent 
        # the gradient from propagating back in time
        # (Setting to none causes them to be reinitialized)
        self.current_fixation = None 
        self.buffer = None

    def get_default_fovea_size(self):
        return self.foveation_module.crop_width,self.foveation_module.crop_height

    def forward(self, current_image):
        """
        Args:
            current_image (torch.tensor): (batch, channels, height, width) image
        """
        # Initialize the current fixation if necessary
        if self.current_fixation is None:
            self.batch_size = current_image.shape[0]
            self.current_fixation = torch.ones((self.batch_size, self.fixation_length), dtype=torch.float32, device=current_image.device)*0.5

        # Extract features from the peripheral image
        background_img = self.downsampler(current_image)
        peripheral_feature = self.peripheral_model(background_img)
        logging.debug(f"Peripheral feature shape: {peripheral_feature.shape}")

        # Extract features from the foveal patch
        foveal_patch = self.foveation_module(
            self.current_fixation, current_image
        )
        logging.debug(f"Foveal patch shape: {foveal_patch.shape}")
        foveal_feature = self.foveal_model(foveal_patch)
        logging.debug(f"Foveal feature shape: {foveal_feature.shape}")
        logging.debug(f"Current fixation shape: {self.current_fixation.shape}")

        # Add the current fixation to the buffer
        all_features = torch.cat([peripheral_feature, foveal_feature, self.current_fixation], dim=1)
        logging.debug(f"All features shape: {all_features.shape}")

        # Initialize the buffer if necessary 
        if self.buffer is None:
            self.buffer = (torch.rand_like(all_features.unsqueeze(1).expand(-1,self.buffer_len,-1), dtype=torch.float32, device=current_image.device)-0.5)*0.1
        
        temp_buffer = torch.cat([all_features.unsqueeze(1), self.buffer], dim=1)
        logging.debug(f"Temp buffer shape: {temp_buffer.shape}")
        self.buffer = temp_buffer[:, :self.buffer_len, :]
        logging.debug(f"Buffer shape: {self.buffer.shape}")

        bbox, next_fixation = self.combiner_model(self.buffer)
        self.current_fixation = next_fixation.detach()
        return bbox, next_fixation 


if __name__ == "__main__":
    batch_size=5
    test_input = torch.randn(batch_size, 3, 224, 224)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Test input shape: {test_input.shape}")
    model = PeripheralFovealVisionModel()
    logging.info("Model summary:")
    logging.info(summary(model))

    bbox, fixation = model(test_input)
    logging.info(f"Output from test input:")
    logging.info(f"    Current bbox shape {bbox.shape}")
    logging.info(f"    bbox: {bbox}")
    logging.info(f"    Current fixation shape {fixation.shape}")
    logging.info(f"    fixation: {fixation}")

