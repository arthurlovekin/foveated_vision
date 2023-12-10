import logging
import math

import torch
from torch import nn
from torchinfo import summary
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Resize

from foveation_module import FoveationModule

RESNET_DEFAULT_INPUT_SIZE = (224, 224)

class EmbeddingModel(nn.Module):
    """
    Creates a feature vector from an image using a pre-trained ResNet50 model.
    """

    def __init__(self):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        if hasattr(self.pretrained, "fc"):
            # self.pretrained.fc = torch.nn.Identity()
            self.pretrained.fc = nn.Linear(2048, 2048)
        else:
            logging.error("No fc layer found in pretrained model")
        # self.pretrined_new = torch.nn.Sequential(*list(self.pretrained.children())[:-1])
        # Freeze the weights of the pretrained model except for the last layer

    def forward(self, low_res_image):
        # output: (batch, 2048) feature vector
        return self.pretrained(low_res_image)

class SingleFrameTrackingModel(nn.Module):
    """
    Takes in the previous image, previous bounding box, and current image.
    Outputs a new bounding box for the current image.
    At train time, the previous bounding box will be the ground truth.
    At test time, the previous bounding box will be the predicted bounding box.
    Model will use a CNN to extract features from the images, and then a
    fully connected layer to predict the new bounding box.
    """
    
    def __init__(self, input_size=RESNET_DEFAULT_INPUT_SIZE):
        super().__init__()
        self.embedding_model = EmbeddingModel()
        self.resize = Resize(input_size, antialias=True)
        self.embedding_size = 2048
        self.fc = nn.Sequential(
            nn.Linear(2*self.embedding_size+4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.ReLU(),
        )

    def forward(self, prev_image, prev_bounding_box, curr_image):
      """
      Predicts the new bounding box for the current image.
      """
      prev_embedding = self.embedding_model(self.resize(prev_image))
      curr_embedding = self.embedding_model(self.resize(curr_image))
      # Concatenate the embeddings and the previous bounding box
      logging.debug("prev_embedding.shape: ", prev_embedding.shape)
      logging.debug("curr_embedding.shape: ", curr_embedding.shape)
      logging.debug("prev_bounding_box.shape: ", prev_bounding_box.shape)
      embedding = torch.cat((prev_embedding, curr_embedding, prev_bounding_box), dim=1)
      logging.debug("embedding.shape: ", embedding.shape)
      # Predict the new bounding box
      new_bounding_box = self.fc(embedding)
      return new_bounding_box

if __name__ == "__main__":
    batch_size = 5
    test_input = torch.randn(batch_size, 3, 512, 512)
    test_labels = torch.randn(batch_size, 4)
    test_curr_image = torch.randn(batch_size, 3, 512, 512)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Test input shape: {test_input.shape}")
    model = SingleFrameTrackingModel()
    logging.info("Model summary:")
    logging.info(summary(model))

    bbox = model(test_input, test_labels, test_curr_image)
    logging.info(f"bbox.shape: {bbox.shape}")

