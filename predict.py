import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_label_to_int = {
    'building-no-damage': 0,
    'building-unknown': 1,
    'building-un-classified': 2,
    'building-minor-damage': 3,
    'building-major-damage': 4,
    'building-destroyed': 5
    # Add more class labels and corresponding integer values as needed
}

# Load the trained model
model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
num_classes = 7  # +1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('object_detection_model.pth', map_location=torch.device('cpu')))
model.eval()
model.to(device)  # Move the model to the same device used during training

test_image_path = 'maraş2.png'
test_image = Image.open(test_image_path).convert('RGB')

# Apply the same preprocessing transformations as during training
new_size = (224, 224)
test_image = F.resize(test_image, new_size)
test_image_tensor = F.to_tensor(test_image)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
test_image_tensor = F.normalize(test_image_tensor, mean, std)
test_image_tensor = test_image_tensor.to(device)  # Move the image tensor to the same device as the model

# Make predictions
with torch.no_grad():
    pred = model([test_image_tensor])

# Get the predicted bounding boxes and labels
predicted_boxes = pred[0]['boxes'].cpu()
predicted_labels = pred[0]['labels'].cpu()

# Draw the predicted bounding boxes on the image

fig, ax = plt.subplots(1)
ax.imshow(test_image)

for box, label in zip(predicted_boxes, predicted_labels):
    box = box.tolist()
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)


    class_label = label.item()
    ax.text(box[0], box[1], class_label, color='r')

plt.show()
