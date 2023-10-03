import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch_device_type import get_device

device = torch.device(get_device())


int_to_color = ["green", "yellow", "blue", "orange", "red", "white", "black"]

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
new_size = (1024, 1024)
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
    class_label = label.item()

    print(box, class_label)

    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=int_to_color[class_label],
                             facecolor='none')
    ax.add_patch(rect)

    # ax.text(box[0], box[1], class_label, color=int_to_color[class_label])

plt.show()
