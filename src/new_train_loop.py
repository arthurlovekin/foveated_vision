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

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = get_dataloader(batch_size=batch_size_train, targ_size=(224, 224), clip_length_s=clip_length_s_train)
test_loader = get_dataloader(batch_size=batch_size_test, targ_size=(224, 224), clip_length_s=clip_length_s_test)

model = PeripheralFovealVisionModel()
model = model.to(device)
# model = torchvision.models.resnet50(pretrained=True)  # For testing before model is ready

# Parallelize training across multiple GPUs
# model = torch.nn.DataParallel(model)

foveation_loss = PeripheralFovealVisionModelLoss()
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

for epoch in range(num_epochs):
    running_loss = 0.0 # for printing 

    # Loop through all (batches of) video sequences
    for j, (seq_inputs, seq_labels) in enumerate(train_loader):
        seq_loss = 0.0

        # Zero out the optimizer
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        # Need to detach then reattach the hidden variables of the model 
        # (Setting to none causes them to be reinitialized)
        model.buffer = None 
        model.current_fixation = None

        # Move to GPU
        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)
        # logging.info(f"Current sequence shape: {seq_inputs.shape}")   # B,Frames,C,W,H
        # logging.info(f"Current label list shape: {seq_labels.shape}") # B,Frames,4

        # Iterate through the sequence of frames and accumulate the loss 
        num_frames = seq_inputs.shape[1]
        for i in range(num_frames-1): # -1 because the loss depends on the next frame as well
            curr_frame = seq_inputs[:,i,:,:,:]  # Batch, Frames, C,W,H
            curr_label = seq_labels[:,i,:]      
            next_label = seq_labels[:,i+1,:]
        
            #logging Memory
            if torch.cuda.is_available():
                used_memory = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"Current used CUDA memory: {used_memory} GB")

            # Run on the "current" frame to generate fixation for the "next" inputs (popped in the current iteration)
            curr_bbox, next_fixation = model(curr_frame)

            # Add to the loss (We'll backprop once the whole video is done)
            seq_loss += foveation_loss(curr_bbox, next_fixation, curr_label, next_label)

        # Update the weights
        seq_loss.backward()
        optimizer.step()

        # Log training info
        running_loss += seq_loss.item()
        batches_between_prints = 1
        if j % batches_between_prints == 0:
            last_loss = running_loss/batches_between_prints
            logging.info(f"Batch {j+1} loss: {last_loss}")
            # writer.add_scalar('Loss/train', total_loss / total_samples, epoch)  # Average loss
            running_loss = 0.0
        
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss / total_samples:.4f}")
print(f"\nFinished training, Loss: {total_loss / total_samples:.4f}")