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
logging.basicConfig(level=logging.INFO)  # Change this to INFO or WARNING to reduce verbosity, or DEBUG for max spam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3
batch_size_train = 2
batch_size_test = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 10
clip_length_s_train = 0.5
clip_length_s_test = 1
save_model = True
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
save_frequency = 100  # Save model every N steps

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = get_dataloader(batch_size=batch_size_train, targ_size=(224, 224), clip_length_s=clip_length_s_train)
test_loader = get_dataloader(batch_size=batch_size_test, targ_size=(224, 224), clip_length_s=clip_length_s_test)

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

    def __len__(self):
        return self.num_frames


for epoch in tqdm(range(num_epochs)):
    epoch_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)
    step = 0
    for seq_inputs, seq_labels in epoch_progress_bar:
        total_loss = 0.0
        total_samples = 0

        # Zero out the optimizer
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        # Need to detach then reattach the hidden variables of the model to prevent 
        # the gradient from propagating back in time
        # (Setting to none causes them to be reinitialized)
        model.buffer = None
        model.fixation = None

        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)
        # Each iteration is a batch of sequences of images
        frame = 0
        num_frames = seq_inputs.shape[1] 
        curr_inputs = None  # Train on these so we have access to the "next" fixation
        curr_labels = None
        # Iterate through the sequence and train on each one
        seq_iterator = SequenceIterator(seq_inputs, seq_labels)
        frame_progress_bar = tqdm(seq_iterator, total=num_frames, desc=f"Step {step+1} progress", position=1, leave=None)
        for inputs, labels in frame_progress_bar: 
            # Each iteration is a batch of sequences of images
            # Iterate through the sequence and train on each one
            if curr_inputs is None or curr_labels is None:
                # Already on device as views of larger tensors
                curr_inputs = inputs
                curr_labels = labels
                frame += 1
                continue
            
            #logging
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

        
        epoch_progress_bar.set_postfix(
            loss=f"{total_loss.item() / total_samples:.4f}",
        )
        # Update the weights
        # Make sure loss values don't depend on batch size or frame count
        total_loss = total_loss / total_samples
        # Calculate the gradient of the accumulated loss only at the end of the loop (not inside)
        total_loss.backward()
        optimizer.step()
        step += 1

        # Log training info
        writer.add_scalar('Loss/train', total_loss, step)  # Average loss

        # TODO: Evaluate on test set

        # Save model checkpoint
        if save_model and step % save_frequency == 0:
            model_path = os.path.join(model_dir, f"model_checkpoint_epoch_{epoch+1}_step_{step}.pth")
            torch.save(model.state_dict(), model_path)

    epoch_progress_bar.close()
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss:.4f}")

def test(model, test_loader, loss_fn):
    # Evaluate on Test set https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f"Test loss: {avg_vloss:.4f}")

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    {'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # # Track best performance, and save the model's state
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     # model_path = 'model_{}_{}'.format(timestamp, epoch)
    #     # torch.save(model.state_dict(), model_path)
    
    return avg_vloss

print(f"\nFinished training, Loss: {total_loss:.4f}")