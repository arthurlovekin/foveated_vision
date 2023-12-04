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
    # Loop through all (batches of) video sequences
    for seq_inputs, seq_labels in train_loader:
        seq_loss = 0.0
        seq_samples_processed = 0

        # Zero out the optimizer
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        # Move to GPU
        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)

        curr_frame = None  # Train on these so we have access to the "next" fixation
        curr_label = None

        # Iterate through the sequence of frames and accumulate the loss 
        n_frames = seq_inputs.shape[1]
        for i in range(n_frames-1): # -1 because the loss depends on the next frame as well
            curr_frame = seq_inputs[:,i,:,:,:] # B,F,C,W,H
            curr_label = seq_inputs[:,i,:] # B,F,4
            next_label = seq_inputs[:,i+1,:]
        
            #logging
            logging.debug(f"Current frame input shape: {curr_inputs.shape}")
            logging.debug(f"Current frame label shape: {curr_labels.shape}")
            if torch.cuda.is_available():
                used_memory = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"Current used CUDA memory: {used_memory} GB")
                writer.add_scalar('Memory/CUDA_used_GiB', used_memory, step*num_frames + frame)

            # Run on the "current" frame to generate fixation for the "next" inputs (popped in the current iteration)
            curr_bbox, next_fixation = model(curr_inputs)
            logging.debug(f"Current estimated bbox: {curr_bbox}")
            logging.debug(f"Next fixation: {next_fixation}")
            loss = foveation_loss(curr_bbox, next_fixation, curr_labels, next_labels)
            loss = loss/num_frames

            # Backward pass to accumulate gradients
            # https://stackoverflow.com/questions/53331540/accumulating-gradients
            
            # if seq_iterator.frame == len(seq_iterator)-1:
            #     loss.backward()
            # else:
            #     loss.backward(retain_graph=True)
            # writer.add_scalar('Loss/train_frame', loss.detach(), step*num_frames + frame)  # Loss for each frame

            # TODO: are these correct/meaningful?
            total_loss += loss
            total_samples += curr_labels.shape[0]

            # Save current frame for next iteration
            curr_inputs = next_inputs
            curr_labels = next_labels
            frame += 1

            # Free up memory. Must be done manually? https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
            # del loss, curr_bbox, next_fixation
        
        epoch_progress_bar.set_postfix(
            loss=f"{total_loss.item() / total_samples:.4f}",
        )
        # Update the weights
        total_loss.backward()
        optimizer.step()
        logging.info('stepped')
        step += 1

        # Log training info
        writer.add_scalar('Loss/train', total_loss / total_samples, epoch)  # Average loss

        # Free up memory. Must be done manually? https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        # del seq_inputs, seq_labels, seq_iterator
        # TODO: Evaluate on test set
        # TODO: Save checkpoint

    epoch_progress_bar.close()
    print(f"\nFinished epoch {epoch+1}/{num_epochs}, loss: {total_loss / total_samples:.4f}")
print(f"\nFinished training, Loss: {total_loss / total_samples:.4f}")