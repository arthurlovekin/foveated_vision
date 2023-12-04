# From https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# Simple example training ResNet on MNIST just as a proof of concept test for training.
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataset.vot_dataset import *
from dataset.got10k_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel
from loss_functions import PeripheralFovealVisionModelLoss, IntersectionOverUnionLoss
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Change this to INFO or WARNING to reduce verbosity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3
batch_size_train = 4
batch_size_test = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = get_dataloader(batch_size=batch_size_train, targ_size=(224, 224))
test_loader = get_dataloader(batch_size=batch_size_test, targ_size=(224, 224))

# Load the model
model = PeripheralFovealVisionModel()
# model = torchvision.models.resnet50(pretrained=True)  # For testing before model is ready

# Parallelize training across multiple GPUs
# model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

foveation_loss = PeripheralFovealVisionModelLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set up Tensorboard logging
writer = SummaryWriter()
# Show a batch of images
images, labels = next(iter(train_loader))
# images = images[0, :, :, :, :]  # Remove the batch dimension so we can display an entire sequence
images = images[:, 0, :, :, :]  # Remove the sequence dimension so we can display the first frame for an entire batch
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
# writer.add_graph(model, images) # TODO: Fix "RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient"

# Train the model...
print(f"Starting training with {num_epochs} epochs, batch size of {batch_size_train}, learning rate {learning_rate}, on device {device}")
model.zero_grad()

class SequenceIterator:
    """
    Iterate through the sequence but keep the batch dimension.
    Iterator class allows us to use tqdm progress bar.
    """
    def __init__(self, seq_inputs, seq_labels):
        """
        Args:
            seq_inputs (torch.tensor): (batch, seq_len, channels, height, width) image
            seq_labels (torch.tensor): (batch, seq_len, 4) bounding box
        """
        self.seq_inputs = seq_inputs
        self.seq_labels = seq_labels
        self.frame = 0
        self.num_frames = seq_inputs.shape[1]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.frame >= self.num_frames:
            raise StopIteration
        inputs = self.seq_inputs[:, self.frame, :, :, :]
        labels = self.seq_labels[:, self.frame, :]
        self.frame += 1
        return inputs, labels

total_loss = 0.0
total_samples = 0
for epoch in tqdm(range(num_epochs)):
    epoch_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for seq_inputs, seq_labels in epoch_progress_bar:
        # Zero out the optimizer
        optimizer.zero_grad()
        model.zero_grad()

        # Each iteration is a batch of sequences of images
        frame = 0
        num_frames = len(seq_inputs)
        curr_inputs = None  # Train on these so we have access to the "next" fixation
        curr_labels = None
        # Iterate through the sequence and train on each one
        seq_iterator = SequenceIterator(seq_inputs, seq_labels)
        frame_progress_bar = tqdm(seq_iterator, desc=f"Step {frame+1}/{num_frames}", leave=False)
        for inputs, labels in frame_progress_bar: 
            # Each iteration is a batch of sequences of images
            # Iterate through the sequence and train on each one
            if curr_inputs is None or curr_labels is None:
                curr_inputs = inputs.to(device)
                curr_labels = labels.to(device)
                continue
            
            logging.debug(f"Current frame input shape: {curr_inputs.shape}")
            logging.debug(f"Current frame label shape: {curr_labels.shape}")

            next_inputs = inputs.to(device)
            next_labels = labels.to(device)
            # Forward pass
            # Run on the "current" frame to generate fixation for the "next" inputs (popped in the current iteration)
            curr_bbox, next_fixation = model(curr_inputs)
            loss = foveation_loss(curr_bbox, next_fixation, curr_labels, next_labels) # TODO: Also take in the fixation point
            loss = loss/num_frames

            # Backward pass to accumulate gradients
            # https://stackoverflow.com/questions/53331540/accumulating-gradients
            loss.backward()

            # TODO: are these correct/meaningful?
            total_loss += loss.item()
            total_samples += curr_labels.shape[0]

            # Save current frame for next iteration
            curr_inputs = next_inputs
            curr_labels = next_labels
            frame += 1

        epoch_progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
        )

        # Update the weights
        optimizer.step()

        # Log training info
        writer.add_scalar('Loss/train', loss, epoch)

        # TODO: Evaluate on test set
        # TODO: Save checkpoint

    epoch_progress_bar.close()
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss / total_samples:.4f}")
print(f"\nFinished training, Loss: {total_loss / total_samples:.4f}")