# From https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# Simple example training ResNet on MNIST just as a proof of concept test for training.
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataset.vot_dataset import *
from dataset.got10k_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel
from loss_functions import PeripheralFovealVisionModelLoss, IntersectionOverUnionLoss
import tqdm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3
batch_size_train = 4
batch_size_test = 100
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

foveation_loss = IntersectionOverUnionLoss() #PeripheralFovealVisionModelLoss()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set up Tensorboard logging
writer = SummaryWriter()
# Show a batch of images
images, labels = next(iter(train_loader))
images = images[0, :, :, :, :]  # Remove the batch dimension so we can display
print(images.shape)
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)

# Train the model...
print(f"Starting training with {num_epochs} epochs, batch size of {batch_size_train}, learning rate {learning_rate}, on device {device}")
for epoch in tqdm(range(num_epochs)):
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for input, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        print(f"Input shape: {inputs.shape}")
        print(f"Label shape: {labels.shape}")

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs) # TODO: outputs will be a tuple of (bbox, fixation)
        loss = foveation_loss(outputs, labels) # TODO: Also take in the fixation point
        # # loss = iou_loss(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_samples += labels.shape[0]

        # Log training info
        writer.add_scalar('Loss/train', loss, epoch)

        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
        )

    progress_bar.close()
    print(f"Finished epoch {epoch+1}/{num_epochs}, loss: {total_loss / total_samples:.4f}")
    # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print(f"Finished training, Loss: {total_loss / total_samples:.4f}")
# print(f'Finished Training, Loss: {loss.item():.4f}')