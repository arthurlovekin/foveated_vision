# From https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# Simple example training ResNet on MNIST just as a proof of concept test for training.

# fmt: off
import torchvision
torchvision.disable_beta_transforms_warning() # Must be called before importing torchvision.transforms anywhere
# fmt: on

import torch

import logging

import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.got10k_dataset import *
from dataset.vot_dataset import *
from loss_functions import SimpleMseLoss 
from single_frame_model import SingleFrameTrackingModel
from utils import bbox_to_img_coords, make_bbox_grid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO
)  # Change this to INFO or WARNING to reduce verbosity, or DEBUG for max spam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5
batch_size_train = 16  # Also used for test
learning_rate = 5e-6
clip_length_s_train = 0.09  # Two frames
clip_length_s_test = 0.5
save_model = True
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
test_frequency = 50  # Evaluate on test set every N steps.
save_frequency = (
    test_frequency * 5
)  # Save model every N steps. Must be a multiple of test_frequency as we only save if the test loss is better.
use_epoch_progress_bar = (
    True  # Use epoch progress bar in addition to step progress bar
)

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Set up Tensorboard logging
writer = SummaryWriter(flush_secs=2)

# train_loader = get_dataloader(
#     batch_size=batch_size_train, targ_size=(224, 224), clip_length_s=clip_length_s_train,
# )
# test_loader = get_dataloader(
#     batch_size=batch_size_test, targ_size=(224, 224), clip_length_s=clip_length_s_test)
seed = int(time.time()*1000)
# Save seed to tensorboard
writer.add_scalar("Dataloader/seed", seed, 0)
train_loader, test_loader = get_train_test_dataloaders(batch_size=batch_size_train,targ_size=(512, 512),clip_length_s=clip_length_s_train, seed=seed)

# Load the model
model = SingleFrameTrackingModel(input_size=(512, 512))

# Parallelize training across multiple GPUs
# model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

loss_fn = SimpleMseLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Show a batch of images
images, labels = next(iter(train_loader))
# images = images[0, :, :, :, :]  # Remove the batch dimension so we can display an entire sequence
images = images[
    :, 0, :, :, :
]  # Remove the sequence dimension so we can display the first frame for an entire batch
grid = torchvision.utils.make_grid(images)
writer.add_image("images", grid, 0)
# writer.add_graph(model, images) # TODO: Fix "RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient"

# Train the model...
print(
    f"Starting training with {num_epochs} epochs, batch size of {batch_size_train}, learning rate {learning_rate}, on device {device}"
)
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
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            # Process pair
            curr_inputs = vinputs[:, 0, :, :, :]
            curr_labels = vlabels[:, 0, :]
            bboxes = []  # For tensorboard visualization.
            gt_bboxes = []  # For tensorboard visualization.
            images = []  # For tensorboard visualization.
            # TODO: show bounding box on image in tensorboard
            total_samples += curr_labels.shape[0]
            next_inputs = vinputs[:, 1, :, :, :]
            next_labels = vlabels[:, 1, :]
            pred_bbox = model(curr_inputs, curr_labels, next_inputs)
            # Add the bounding box to the list for visualization
            bboxes.append(pred_bbox)
            gt_bboxes.append(next_labels)
            images.append(next_inputs)
            vloss = loss_fn(pred_bbox, next_labels)
            running_vloss += vloss
            # Create a grid of images with bounding boxes
            # For now, just show the first clip in the batch
            logging.info(f"Test loop: first estimated bbox: {bboxes[0]}")
            try:
                bbox_grid = make_bbox_grid(images, bboxes, gt_bboxes, decimation=1)
                writer.add_image("images/test", bbox_grid, step)
                logging.info(f"Wrote image grid to tensorboard at step {step}")
            except Exception as e:
                # This can happen if the bounding boxes are so far off tensorboard
                # doesn't recognize them as the valid format.
                logging.error(f"Error creating image grid: {e}")
            writer.flush()  # Necessary, otherwise tensorboard doesn't update
            break  # Just do one batch for now, otherwise it'd take forever?
    avg_vloss = (
        running_vloss / total_samples
    )  # Divide by total number of frames sampled across all batches
    print(f"Test loss: {avg_vloss.item():.4f}")

    # Log the running loss averaged per batch
    writer.add_scalar("Loss/validation", avg_vloss, step)
    writer.flush()  # Unnecessary?
    return avg_vloss


best_test_loss = float("inf")
step = 0
for epoch in range(num_epochs):
    logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
    epoch_progress_bar = None
    if use_epoch_progress_bar:
        epoch_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True
        )  # Update position of other bars if using.
    for seq_inputs, seq_labels in epoch_progress_bar:
        total_loss = 0.0
        total_samples = 0

        # Zero out the optimizer
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)
        curr_inputs = seq_inputs[:, 0, :, :, :]
        curr_labels = seq_labels[:, 0, :]
        next_inputs = seq_inputs[:, 1, :, :, :]
        next_labels = seq_labels[:, 1, :]
        logging.debug(f"Current frame input shape: {curr_inputs.shape}")
        logging.debug(f"Current frame label shape: {curr_labels.shape}")
        if torch.cuda.is_available():
            # logging.info(torch.cuda.memory_summary(device=device, abbreviated=False))  # Very verbose
            # Get free CUDA memory in GiB
            used_memory = torch.cuda.memory_allocated() / 1024**3
            logging.debug(f"Current used CUDA memory: {used_memory}")
            writer.add_scalar(
                "Memory/CUDA_used_GiB", used_memory, step
            )

        # Forward pass
        # Run on the "current" frame to generate fixation for the "next" inputs (popped in the current iteration)
        pred_bbox = model(curr_inputs, curr_labels, next_inputs)
        logging.debug(f"Current estimated bbox: {pred_bbox}")
        logging.debug(f"Currrent true bbox: {curr_labels}")
        logging.debug(f"Next true bbox: {next_labels}")
        total_loss = loss_fn(pred_bbox, next_labels)
        total_samples = curr_labels.shape[0]
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
        writer.add_scalar("Loss/train", total_loss, step)  # Average loss
        # Evaluate on test set
        if step % test_frequency == 0:
            test_loss = test(model, test_loader, loss_fn, step).item()
            better = test_loss < best_test_loss
            logging.info(f"New test loss: {test_loss:.4f}; best test loss: {best_test_loss:.4f}; better: {better}")
            if better and save_model and step % save_frequency == 0 and step != 0:
                logging.info(f"Saving model.")
                best_test_loss = test_loss
                # Save model checkpoint
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(
                    model_dir, f"{date_str}_model_epoch_{epoch+1}_step_{step}_{best_test_loss:.6f}.pth"
                )
                torch.save(model.state_dict(), model_path)
            model.train()  # Set back to train mode
        step += 1

    if use_epoch_progress_bar:
        epoch_progress_bar.close()
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss:.4f}")


print(f"\nFinished training, Loss: {total_loss:.4f}")
