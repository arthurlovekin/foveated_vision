# https://pytorch.org/vision/stable/models.html
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

# from pathlib import Path
# img_filepath = Path("src/demo/abbeyroad.jpg")
# img = read_image(str(img_filepath))
img = read_image("src/demo/abbeyroad.jpg")

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")