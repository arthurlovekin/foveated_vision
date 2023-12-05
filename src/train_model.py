# From https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# Simple example training ResNet on MNIST just as a proof of concept test for training.
from datetime import datetime
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from dataset.vot_dataset import *
from dataset.got10k_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel
from loss_functions import PeripheralFovealVisionModelLoss, IntersectionOverUnionLoss
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change this to INFO or WARNING to reduce verbosity, or DEBUG for max spam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3
batch_size_train = 3
batch_size_test = 10
learning_rate = 0.01
momentum = 0.5
clip_length_s_train = 0.5
clip_length_s_test = 1
save_model = True
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
save_frequency = 100  # Save model every N steps
test_frequency = 10  # Evaluate on test set every N steps
use_epoch_progress_bar = False  # Use epoch progress bar in addition to step progress bar

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = get_dataloader(batch_size=batch_size_train, targ_size=(224, 224), clip_length_s=clip_length_s_train)
test_loader = get_dataloader(batch_size=batch_size_test, targ_size=(224, 224), clip_length_s=clip_length_s_test)

# Load the model
model = PeripheralFovealVisionModel()

# Parallelize training across multiple GPUs
# model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

foveation_loss = PeripheralFovealVisionModelLoss()
foveation_loss.foveation_weight = 0.0  # TODO: Remove this, just for testing
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set up Tensorboard logging
writer = SummaryWriter(flush_secs=2)
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

    def __len__(self):
        return self.num_frames

def bbox_to_img_coords(bbox, image):
    """ Convert a bounding box from [0, 1] range to image coordinates
    Also ensure that the bounding box is within the image bounds and
    clip if necessary.
    Args:
        bbox (torch.tensor): (batch, 4) bounding box
        image (torch.tensor): (channels, height, width) image
    """
    # Convert each corner and clip to image bounds
    bbox[:, 0] = torch.clamp(bbox[:, 0] * image.shape[2], min=0, max=image.shape[2])
    bbox[:, 1] = torch.clamp(bbox[:, 1] * image.shape[1], min=0, max=image.shape[1])
    bbox[:, 2] = torch.clamp(bbox[:, 2] * image.shape[2], min=0, max=image.shape[2])
    bbox[:, 3] = torch.clamp(bbox[:, 3] * image.shape[1], min=0, max=image.shape[1])
    return bbox

def make_bbox_grid(images, bboxes):
    """ 
    Create a grid of images with bounding boxes
    Args:
        images (list): List of images, each of shape (batch, channels, height, width)
        bboxes (list): List of bounding boxes, each of shape (batch, 4)
    """
    # For now, just show the first clip in the batch
    batch_ind = 0
    bbox_list = []
    for i in range(len(images)):
        image = TF.convert_image_dtype(images[i][batch_ind, :, :, :], dtype=torch.uint8)
        # Requires a dimension to possibly display multiple bounding boxes
        bbox = bboxes[i][batch_ind, :].unsqueeze(0)
        # Convert bounding boxes from [0, 1] range to image coordinates
        bbox = bbox_to_img_coords(bbox, image) # Also clips to image dimensions
        bbox_list.append(torchvision.utils.draw_bounding_boxes(image, bbox, colors=["green"], width=5))
    bbox_grid = torchvision.utils.make_grid(bbox_list)
    return bbox_grid

def test(model, test_loader, loss_fn, step=0):
    # Evaluate on Test set https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    running_vloss = 0.0
    total_samples = 0.0
    model.eval()
    with torch.no_grad():
        # TODO: can we just evaluate on some of the test set?
        # Set up the progress bar
        logging.info(f"Evaluating on test set...")
        # progress_bar = tqdm(test_loader, desc=f"Test set progress", position=0, leave=True)  # Use for tqdm bar on entire test set
        for i, vdata in enumerate(test_loader):
            model.reset()  # Reset the hidden state
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            # Iterate through the sequence and test on each one
            seq_iterator = SequenceIterator(vinputs, vlabels)
            curr_inputs = None  # Evalulate on these so we have access to the "next" fixation
            curr_labels = None
            frame_progress_bar = tqdm(seq_iterator, total=vinputs.shape[1], desc=f"Test set progress", position=1, leave=True)
            bboxes = []  # For tensorboard visualization.
            images = []  # For tensorboard visualization.
            # TODO: show bounding box on image in tensorboard
            for inputs, labels in frame_progress_bar:
                if curr_inputs is None or curr_labels is None:
                    # Already on device as views of larger tensors
                    curr_inputs = inputs
                    curr_labels = labels
                    continue
                total_samples += curr_labels.shape[0]
                next_inputs = inputs
                next_labels = labels
                curr_bbox, next_fixation = model(curr_inputs)
                logging.info(f"Test loop: Current estimated bbox: {curr_bbox}")
                # Add the bounding box to the list for visualization
                images.append(curr_inputs)
                bboxes.append(curr_bbox)
                vloss = loss_fn(curr_bbox, next_fixation, curr_labels, next_labels)
                running_vloss += vloss
                curr_inputs = next_inputs
                curr_labels = next_labels
            # Create a grid of images with bounding boxes
            # For now, just show the first clip in the batch
            bbox_grid = make_bbox_grid(images, bboxes)
            writer.add_image('images/test', bbox_grid, step)
            logging.info(f"Wrote image grid to tensorboard at step {step}")
            writer.flush()  # Necessary, otherwise tensorboard doesn't update
            break  # Just do one batch for now, otherwise it'd take forever?
    avg_vloss = running_vloss / total_samples  # Divide by total number of frames sampled across all batches
    print(f"Test loss: {avg_vloss.item():.4f}")

    # Log the running loss averaged per batch
    writer.add_scalar('Loss/validation',
                    avg_vloss,
                    step)
    writer.flush()  # Unnecessary?
    return avg_vloss


for epoch in range(num_epochs):
    logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
    epoch_progress_bar = None
    if use_epoch_progress_bar:
        epoch_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)  # Update position of other bars if using.
    step = 0
    for seq_inputs, seq_labels in train_loader:
        total_loss = 0.0
        total_samples = 0

        # Zero out the optimizer
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model.reset()  # Reset the hidden state

        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)
        # Each iteration is a batch of sequences of images
        frame = 0
        num_frames = seq_inputs.shape[1] 
        curr_inputs = None  # Train on these so we have access to the "next" fixation
        curr_labels = None
        # Iterate through the sequence and train on each one
        seq_iterator = SequenceIterator(seq_inputs, seq_labels)
        frame_progress_bar = tqdm(seq_iterator, total=num_frames, desc=f"Step {step+1} progress",
                                  position=(1 if use_epoch_progress_bar else 0), leave=True)
        for inputs, labels in frame_progress_bar: 
            # Each iteration is a batch of sequences of images
            # Iterate through the sequence and train on each one
            if curr_inputs is None or curr_labels is None:
                # Already on device as views of larger tensors
                curr_inputs = inputs
                curr_labels = labels
                frame += 1
                continue
            
            logging.debug(f"Current frame input shape: {curr_inputs.shape}")
            logging.debug(f"Current frame label shape: {curr_labels.shape}")
            if torch.cuda.is_available():
                # logging.info(torch.cuda.memory_summary(device=device, abbreviated=False))  # Very verbose
                # Get free CUDA memory in GiB
                used_memory = torch.cuda.memory_allocated() / 1024**3
                logging.debug(f"Current used CUDA memory: {used_memory}")
                writer.add_scalar('Memory/CUDA_used_GiB', used_memory, step*num_frames + frame)

            # Already on device as views of larger tensors
            next_inputs = inputs
            next_labels = labels
            # Forward pass
            # Run on the "current" frame to generate fixation for the "next" inputs (popped in the current iteration)
            curr_bbox, next_fixation = model(curr_inputs)
            logging.debug(f"Current estimated bbox: {curr_bbox}")
            logging.debug(f"Next fixation: {next_fixation}")
            loss = foveation_loss(curr_bbox, next_fixation, curr_labels, next_labels)
            loss = loss

            writer.add_scalar('Loss/train_frame', loss.detach(), step*num_frames + frame)  # Loss for each frame
            total_loss += loss
            total_samples += curr_labels.shape[0]

            # Save current frame for next iteration
            curr_inputs = next_inputs
            curr_labels = next_labels
            frame += 1
        if use_epoch_progress_bar:
            epoch_progress_bar.set_postfix(
                loss=f"{total_loss.item() / total_samples:.4f}",
            )
        # Update the weights
        # Make sure loss values don't depend on batch size or frame count
        total_loss = total_loss / total_samples
        # Calculate the gradient of the accumulated loss only at the end of the loop (not inside)
        total_loss.backward()
        optimizer.step()

        # Log training info
        writer.add_scalar('Loss/train', total_loss, step)  # Average loss

        # Evaluate on test set
        if step % test_frequency == 0:
            test_loss = test(model, test_loader, foveation_loss, step)
            model.train()  # Set back to train mode

        # Save model checkpoint
        if save_model and step % save_frequency == 0 and step != 0:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"{date_str}_model_epoch_{epoch+1}_step_{step}.pth")
            torch.save(model.state_dict(), model_path)
        step += 1

    if use_epoch_progress_bar:
        epoch_progress_bar.close()
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss:.4f}")


print(f"\nFinished training, Loss: {total_loss:.4f}")