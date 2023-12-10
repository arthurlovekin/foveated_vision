import logging
import math

import torch
from torch import nn
from torchinfo import summary
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Resize

from foveation_module import FoveationModule
from utils import fix_fovea_if_needed

# from typing import Tensor

RESNET_DEFAULT_INPUT_SIZE = (224, 224)


class PeripheralModel(nn.Module):
    """
    Creates a feature vector from a low-resolution background image
    using a pre-trained ResNet50 model. (These features are later combined
    with the foveal features to produce a bounding box and fixation point).
    """

    def __init__(self, frozen=False):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if frozen:
            for param in self.pretrained.parameters():
                param.requires_grad = False
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

    def __init__(self, frozen=False):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if frozen:
            for param in self.pretrained.parameters():
                param.requires_grad = False
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

    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        position_encoding = torch.zeros(max_len, 1, d_model)
        position_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x):
        """x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``"""
        x = x + self.position_encoding[: x.size(0)]
        return self.dropout(x)

class LinearCombiner(nn.Module):
    """
    Takes buffers of foveal features, peripheral features, and fovea points, and
    combines them to produce a bounding box and new fixation point.

    This model flattens the sequence and uses a sequence of linear layers to 
    produce a latent vector. Then, there are two linear heads to predict the bounding box
    and next fixation point.
    """
    
    def __init__(
        self,
        buffer_size=3,
        n_inputs: int = 4098,
        n_object_to_track=2048+4,
    ):
        """
        Args:
            buffer_size (int): Number of previous frames to consider
            n_inputs (int): Number of input features (peripheral features + foveal features + fovea points)
            n_encoder_layers (int): Number of encoder layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.sequence_dim = (buffer_size * n_inputs) + n_object_to_track
        self.encoder = nn.Sequential(
            nn.Linear(self.sequence_dim, self.sequence_dim//2),
            nn.ReLU(),
            nn.Linear(self.sequence_dim//2, self.sequence_dim//2),
            nn.ReLU(),
            nn.Linear(self.sequence_dim//2, self.sequence_dim//2),
            nn.ReLU(),
        )
        self.intermediate_dim = 512  # TODO: Make this a parameter
        self.bbox_head = nn.Sequential(
            nn.Linear(self.sequence_dim//2, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim//4),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim//4, 4),
            nn.Sigmoid(),  # make outputs 0-1
        )
        self.pos_head = nn.Sequential(
            nn.Linear(self.sequence_dim//2, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim//4),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim//4, 2),
            nn.Sigmoid(),  # make outputs 0-1
        )

    def forward(self, all_features_buffer, object_to_track_z):
        # Flatten the sequence
        latent_seq = all_features_buffer.reshape((all_features_buffer.shape[0], -1))
        # Concatenate the object to track to the sequence (early fusion)
        latent_seq = torch.cat([latent_seq, object_to_track_z], dim=1)
        # Run the encoder to get an intermediate representation
        encoded = self.encoder(latent_seq)
        # Predict the bounding box and next fixation point
        bbox = self.bbox_head(encoded)
        pos = self.pos_head(encoded)
        return bbox, pos


class CombinerModel(nn.Module):
    """
    Takes buffers of foveal features, peripheral features, and fovea points, and
    combines them to produce a bounding box and new fixation point.
    """

    def __init__(
        self,
        buffer_size=3,
        n_inputs: int = 4098,
        n_heads: int = 6,
        n_encoder_layers: int = 2,
        dropout: float = 0.1,
        fixation_length=2,
        n_object_to_track=2048+4,
    ):
        """
        Args:
            buffer_size (int): Number of previous frames to consider
            n_inputs (int): Number of input features (peripheral features + foveal features + fovea points)
            n_heads (int): Number of attention heads. Must evenly divide n_inputs*batch_size
            n_encoder_layers (int): Number of encoder layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.positional_encoding = PositionalEncoding(n_inputs, dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_inputs,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, num_layers=n_encoder_layers
        )  # (contains multiple TransformerEncoderLayers)
        # Map the encoded sequence to a bounding box.
        # TODO: This could use a more advanced decoder
        self.sequence_dim = (buffer_size * n_inputs) + n_object_to_track
        self.intermediate_dim = 512  # TODO: Make this a parameter
        self.bbox_head = nn.Sequential(
            nn.Linear(self.sequence_dim, self.intermediate_dim),
            nn.Sigmoid(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim//4),
            nn.Sigmoid(),
            nn.Linear(self.intermediate_dim//4, 4),
            nn.Sigmoid(),  # make outputs 0-1
        )
        self.pos_head = nn.Sequential(
            nn.Linear(self.sequence_dim, self.intermediate_dim),
            nn.Sigmoid(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim//4),
            nn.Sigmoid(),
            nn.Linear(self.intermediate_dim//4, 2),
            nn.Sigmoid(),  # make outputs 0-1
        )
        self.min_bbox_width = 0.01

    def forward(self, all_features_buffer, object_to_track_z):
        # Transformer expects input of shape (batch, seq_len, feature_len)
        # Concatenate all features along the time dimension
        positional_features = self.positional_encoding(all_features_buffer)
        logging.debug(f"Transformer input shape: {positional_features.shape}")
        latent_seq = self.transformer_model(positional_features)
        # Hack decoder
        # Flatten the sequence
        latent_seq = latent_seq.reshape((latent_seq.shape[0], -1))
        # Concatenate the object to track to the sequence
        # TODO(rgg): explore early fusion vs late fusion. Not as helpful for the transformer
        # if this info comes in at the end?
        latent_seq = torch.cat([latent_seq, object_to_track_z], dim=1)
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
        """Outputs both the predicted bounding box,
        and the output attentions of the transformer, which are used by the NeuralFoavationModule to generate a foveal patch.
        """
        pass


class PeripheralFovealVisionModel(nn.Module):
    def __init__(self, use_foveation_module=True):
        self.use_foveation_module = use_foveation_module
        super().__init__()
        self.peripheral_resolution = RESNET_DEFAULT_INPUT_SIZE
        self.downsampler = Resize(
            (self.peripheral_resolution[0], self.peripheral_resolution[1]),
            antialias=True,
        )
        # self.feature_len = 2x2048 (resnet output) + 2 for center of fixation
        # TODO: Just output 4 points (centerx/y, dimensionsx/y) directly for the next fixation instead of 2
        self.fixation_length = 2
        self.buffer_len = 3
        self.foveation_module = FoveationModule(bound_crops=False)
        self.peripheral_model = PeripheralModel()
        self.foveal_model = FovealModel()
        self.combiner_model = CombinerModel(fixation_length=self.fixation_length)
        # self.combiner_model = LinearCombiner(fixation_length=self.fixation_length)
        self.position_encoding = None
        self.current_fixation = None
        self.buffer = None
        # Latent representation of the object to track. Always appended to the buffer
        self.object_to_track_z = None 

    def reset(self):
        # Need to detach then reattach the hidden variables of the model to prevent
        # the gradient from propagating back in time
        # (Setting to none causes them to be reinitialized)
        self.current_fixation = None
        self.buffer = None
        self.object_to_track_z = None

    def initialize(self, current_image, init_bbox):
        """
        Initialize the model with a bounding box and image.
        Should be batch-enabled.
        """
        # Take a high res patch centered at the bbox using the foveation module
        # Get the center of the bbox (batched) (xmin, ymin, xmax, ymax)
        # Need a tensor of shape (batch, 2) for the foveation module
        bbox_center = torch.stack([
                (init_bbox[:, 0] + init_bbox[:, 2]) / 2.0,
                (init_bbox[:, 1] + init_bbox[:, 3]) / 2.0,
            ], dim=1)
        with torch.no_grad():
            patch = self.foveation_module(bbox_center, current_image)
        # Initialize the current fixation to the center of the bbox
        self.current_fixation = bbox_center
        # Run the foveal model on the patch and save the result
        # TODO(rgg): see if we should keep gradients on for this
        with torch.no_grad():
            patch_z = self.foveal_model(patch)
            # Concatenate the bounding box to the patch
            self.object_to_track_z = torch.cat([patch_z, init_bbox], dim=1)
    
    def initialize_buffer(self, current_image, feature_vector):
        """
        Initialize the buffer
        Requeires that the object_to_track_z is already initialized
        """
        # Buffer contains object to track + features from each of the previous frames
        self.buffer = (
            torch.rand_like(
                feature_vector.unsqueeze(1).expand(-1, self.buffer_len, -1),
                dtype=torch.float32,
                device=current_image.device,
            )
            - 0.5
            ) * 0.1  # What are these magic numbers?

    def get_default_fovea_size(self):
        return self.foveation_module.crop_width, self.foveation_module.crop_height

    def forward(self, current_image):
        """
        Args:
            current_image (torch.tensor): (batch, channels, height, width) image
        """
        if len(current_image.shape) == 3: 
            current_image.unsqueeze(0)
        # Initialize the current fixation if necessary
        if self.current_fixation is None:
            self.batch_size = current_image.shape[0]
            self.current_fixation = (
                torch.ones(
                    (self.batch_size, self.fixation_length),
                    dtype=torch.float32,
                    device=current_image.device,
                )
                * 0.5
            )
        if self.object_to_track_z is None:
            # Initialize to the current fixation if not already initialized.
            fovea_bbox = fix_fovea_if_needed(
                self.current_fixation, default_shape=self.get_default_fovea_size()
            )
            self.initialize(current_image, fovea_bbox)
        # Extract features from the peripheral image
        background_img = self.downsampler(current_image)
        peripheral_feature = self.peripheral_model(background_img)
        logging.debug(f"Peripheral feature shape: {peripheral_feature.shape}")

        # Extract features from the foveal patch
        if self.use_foveation_module:
            foveal_patch = self.foveation_module(self.current_fixation, current_image)
        else:
            foveal_patch = current_image
        logging.debug(f"Foveal patch shape: {foveal_patch.shape}")
        foveal_feature = self.foveal_model(foveal_patch)
        logging.debug(f"Foveal feature shape: {foveal_feature.shape}")
        logging.debug(f"Current fixation shape: {self.current_fixation.shape}")

        # Add the current fixation to the buffer
        all_features = torch.cat(
            [peripheral_feature, foveal_feature, self.current_fixation], dim=1
        )
        logging.debug(f"All features shape: {all_features.shape}")

        # Initialize the buffer if necessary
        if self.buffer is None:
            self.initialize_buffer(current_image, all_features)

        temp_buffer = torch.cat([all_features.unsqueeze(1), self.buffer], dim=1)
        logging.debug(f"Temp buffer shape: {temp_buffer.shape}")
        # Take the buffer_len most recent frames.
        self.buffer = temp_buffer[:, :self.buffer_len, :]
        logging.debug(f"Buffer shape: {self.buffer.shape}")
        bbox, next_fixation = self.combiner_model(self.buffer, self.object_to_track_z)
        self.current_fixation = next_fixation
        return torch.squeeze(bbox, dim=0), torch.squeeze(next_fixation, dim=0)


if __name__ == "__main__":
    batch_size = 5
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
