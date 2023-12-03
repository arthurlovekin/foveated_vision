import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule
from torchvision.transforms import Resize
from torchinfo import summary
import math
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
            print("No fc layer found in pretrained model")
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
            print("No fc layer found in pretrained model")

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
                 n_encoder_layers: int=4, dropout: float = 0.1):
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
        # self.transformer_model = nn.Transformer(
        #     nhead=n_heads, num_encoder_layers=n_encoder_layers, num_decoder_layers=0
        # )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=n_inputs, nhead=n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_model = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_encoder_layers) # (contains multiple TransformerEncoderLayers)
        # Map the encoded sequence to a bounding box.
        # This really shouldn't be done like this; we should use a decoder,
        # and the output should be a sequence of bounding boxes (with a loss on each one).
        self.sequence_dim = buffer_size*n_inputs
        self.bbox_head = nn.Linear(self.sequence_dim, 4)  
        self.pos_head = nn.Linear(4, 2)  

    def make_sequence(self, foveal_features, peripheral_features, fovea_points):
        """
        Deprecated?
        Creates a sequence of features from the buffers. Transfomer expects sequence of shape 
        (batch, n, 1) sequence. ??(seq_len, batch, feature_len)??
        """
        return torch.cat(
            [foveal_features, peripheral_features, fovea_points], dim=1
        ).unsqueeze(2)

    # def forward(self, peripheral_features, foveal_features, fovea_points):
    #     input_sequence = self.make_sequence(
    #         foveal_features, peripheral_features, fovea_points
    #     )
    #     return self.transformer_model(input_sequence)
    
    def forward(self, all_features_buffer):
        # Transformer expects input of shape (batch, seq_len, feature_len)
        # Concatenate all features along the time dimension
        positional_features = self.positional_encoding(all_features_buffer)
        print(f"Transformer input shape: {positional_features.shape}")
        latent_seq = self.transformer_model(positional_features)
        # Hack decoder
        # Flatten the sequence
        latent_seq = latent_seq.reshape((latent_seq.shape[0], -1))
        bbox = self.bbox_head(latent_seq)
        pos = self.pos_head(bbox)
        return bbox, pos 


class PeripheralFovealVisionModel(nn.Module):
    def __init__(self):#, batch_size=1):
        super().__init__()
        # self.batch_size = batch_size
        self.peripheral_resolution = RESNET_DEFAULT_INPUT_SIZE
        self.downsampler = Resize(
            (self.peripheral_resolution[0], self.peripheral_resolution[1]),
            antialias=True,
        )
        self.foveation_module = FoveationModule()
        self.peripheral_model = PeripheralModel()
        self.foveal_model = FovealModel()
        self.combiner_model = CombinerModel()
        self.position_encoding = None
        self.fixation_length = 2
        self.current_fixation = None #torch.ones((self.batch_size, self.fixation_length), dtype=torch.float32)*0.5
        self.feature_len = 2048*2+self.fixation_length # 2x resnet output + 2 for center of fixation (TODO: Make this dynamic)
        self.buffer_len = 3
        self.buffer = None #torch.zeros(self.batch_size, self.buffer_len, self.feature_len, dtype=torch.float32)
        
    #     self.foveal_feature_buffer = None
    #     self.peripheral_feature_buffer = None
    #     self.fovea_point_buffer = None
    #     # Initialize the buffers
    #     self.reset_buffers()


    # def reset_buffers(self, batched_image):
    #     # Initialize the buffers with zeros
    #     self.foveal_feature_buffer = torch.zeros(
    #         (self.batch_size, self.buffer_len, self.feature_len), dtype=torch.float32
    #     )
    #     self.fovea_point_buffer = torch.zeros(
    #         (self.batch_size, self.buffer_len, 2), dtype=torch.float32
    #     )
    #     self.peripheral_feature_buffer = torch.zeros(
    #         (self.batch_size, self.buffer_len, self.feature_len), dtype=torch.float32
    #     )

    def forward(self, current_image):
        """
        Args:
            current_image (torch.tensor): (batch, channels, height, width) image
        """
        # Initialize the current fixation if necessary
        if self.current_fixation is None:
            batch_size = current_image.shape[0]
            self.current_fixation = torch.ones((batch_size, self.fixation_length), dtype=torch.float32)*0.5

        # Extract features from the peripheral image
        background_img = self.downsampler(current_image)
        peripheral_feature = self.peripheral_model(background_img)
        print(f"Peripheral feature shape: {peripheral_feature.shape}")

        # Extract features from the foveal patch
        foveal_patch = self.foveation_module(
            self.current_fixation, current_image
        )
        print(f"Foveal patch shape: {foveal_patch.shape}")
        foveal_feature = self.foveal_model(foveal_patch)
        print(f"Foveal feature shape: {foveal_feature.shape}")

        print(f"Current fixation shape: {self.current_fixation.shape}")

        # Add the current fixation to the buffer
        all_features = torch.cat([peripheral_feature, foveal_feature, self.current_fixation], dim=1)
        print(f"All features shape: {all_features.shape}")

        # Initialize the buffer if necessary 
        if self.buffer is None:
            self.buffer = (torch.rand_like(all_features.unsqueeze(1).expand(-1,self.buffer_len,-1), dtype=torch.float32)-0.5)*0.1
        
        temp_buffer = torch.cat([all_features.unsqueeze(1), self.buffer], dim=1)
        print(f"Temp buffer shape: {temp_buffer.shape}")
        self.buffer = temp_buffer[:, :self.buffer_len, :]
        print(f"Buffer shape: {self.buffer.shape}")

        bbox, self.current_fixation = self.combiner_model(self.buffer)

        # Deprecated?: track separate buffers, would be nice to have for debugging
        # # Add the peripheral feature to the buffer
        # self.peripheral_feature_buffer = self.add_vector_to_buffer(
        #     peripheral_feature, self.peripheral_feature_buffer
        # )
        # # Add the foveal feature to the buffer
        # self.foveal_feature_buffer = self.add_vector_to_buffer(
        #     foveal_feature, self.foveal_feature_buffer
        # )

        # # Add the fovea point to the buffer
        # self.fovea_point_buffer = self.add_vector_to_buffer(
        #     self.current_fixation, self.fovea_point_buffer
        # )
        return bbox, self.current_fixation
    
    # def add_to_buffer(self, all_features):
    #     """
    #     Adds a vector to the buffer, popping the oldest vector in the buffer.
    #     Assumes the buffer is of shape (batch, buffer_len, all_feature_len).
    #     ()
    #     """

    #     self.buffer = torch.cat([all_features.unsqueeze(1), self.buffer], dim=1)
    #     return self.buffer[-self.buffer_len :]
    
    # def add_vector_to_buffer(self, vector, buffer):
    #     """
    #     Adds a vector to the buffer, popping the oldest vector in the buffer.
    #     Assumes the buffer is of shape (batch, buffer_len, feature_len).
    #     """
    #     print(f"Adding vector of shape {vector.shape} to buffer of shape {buffer.shape}")
    #     print(f"Unsqueezed vector shape: {vector.unsqueeze(1).shape}")
    #     buffer = torch.cat([vector.unsqueeze(1), buffer], dim=1)
    #     return buffer[-self.buffer_len :]


if __name__ == "__main__":
    # model = PeripheralModel()
    # print("Peripheral model summary:")
    # print(summary(model))
    batch_size=5
    test_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Test input shape: {test_input.shape}")
    model = PeripheralFovealVisionModel()#batch_size=batch_size)
    print("Model summary:")
    print(summary(model))

    bbox, fixation = model(test_input)
    print(f"Output from test input:")
    print(f"    Current bbox shape {bbox.shape}")
    print(f"    Current fixation shape {fixation.shape}")

