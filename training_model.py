import json
import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import glob
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from multiprocessing import Pool
from shapely import wkt
import time

from torch_device_type import get_device


class Progress:
    start_time = None
    prefix = None
    bar_length = None

    def __init__(self, prefix="Progress", bar_length=30, start_immediately=True):
        self.prefix = prefix
        self.bar_length = bar_length
        if start_immediately:
            self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.print(1, 1)
        self.start_time = None
        sys.stdout.write('\n')
        sys.stdout.flush()

    def format_time(self, seconds):
        hours = int(seconds / 3600)
        seconds -= hours * 3600
        minutes = int(seconds / 60)
        seconds -= minutes * 60
        seconds = int(seconds)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def print(self, current, total):
        percent = float(current) * 100 / total
        arrow = '-' * int(percent / 100 * self.bar_length - 1) + '>'
        spaces = ' ' * (self.bar_length - len(arrow))

        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time

        remaining_time = (elapsed_time / (current + 1)) * (total - current)

        sys.stdout.write('\r%s: [%s%s] %d%% | Elapsed: %s | Remaining: %s' %
                         (self.prefix, arrow, spaces, percent, self.format_time(elapsed_time), self.format_time(remaining_time)))
        sys.stdout.flush()


def transform(image, boxes, labels, mean, std):
    # Resize the image and the boxes
    new_size = (1024, 1024)
    image = torchvision.transforms.functional.resize(image, new_size, antialias=True)
    scale_x = new_size[0] / image.size()[1]
    scale_y = new_size[1] / image.size()[0]
    # Divide by scaling factors
    boxes = boxes * torch.tensor([1/scale_x, 1/scale_y, 1/scale_x, 1/scale_y])

    # Normalize the image
    image = torchvision.transforms.functional.normalize(image, mean, std)

    return image, boxes, labels


def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)

    # Create a new list to store the modified targets
    new_targets = []

    # Loop over each target dictionary in the original targets list
    for target_dict in targets:
        # Create a new dictionary to store the modified key-value pairs
        new_target_dict = {}

        # Loop over each key-value pair in the target dictionary
        for key, value in target_dict.items():
            if key == 'boxes':
                # Get the mask from the target dictionary
                mask = target_dict['mask']

                # Use the mask to filter valid bounding boxes
                value = value[mask]
            elif key == 'labels':
                # Get the mask from the target dictionary
                mask = target_dict['mask']

                # Use the mask to filter valid labels
                value = value[mask]

            # Assign the modified value to the same key in the new dictionary
            new_target_dict[key] = value

        # Append the new dictionary to the new list
        new_targets.append(new_target_dict)

    return images, new_targets


class CustomObjectDetectionDataset(Dataset):
    def __init__(self, image_folder, annotations, transform):
        self.image_folder = image_folder
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation, name = self.annotations[index]
        image_name = name
        image_path = os.path.join(self.image_folder, image_name + '.png')
        image = Image.open(image_path).convert('RGB')
        boxes = torch.tensor(annotation[name]['boxes'], dtype=torch.float32)
        labels = torch.tensor(annotation[name]['labels'], dtype=torch.int64)
        image = ToTensor()(image)

        if self.transform is not None:
            mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor(
                [0.229, 0.224, 0.225])
            image, boxes, labels = self.transform(
                image, boxes, labels, mean, std)

        # Set a fixed length for the padding, such as 50
        max_length = 50

        # Create a mask to indicate which elements in the padded tensor are valid
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[:boxes.size(0)] = 1

        targets = {
            'boxes': torch.nn.functional.pad(boxes, (0, 0, 0, max_length - boxes.size(0)), mode='constant',
                                             value=0),
            'labels': torch.nn.functional.pad(labels, (0, max_length - labels.size(0)), mode='constant', value=0),
            'mask': mask
        }

        return image, targets


def parse_json(json_file, data):

    class_label_to_int = {
        'building-no-damage': 0,
        'building-unknown': 1,
        'building-un-classified': 2,
        'building-minor-damage': 3,
        'building-major-damage': 4,
        'building-destroyed': 5
        # Add more class labels and corresponding integer values as needed
    }

    annotations_dict = {}
    for entry in data['features']['xy']:
        properties = entry['properties']
        feature_type = properties['feature_type']
        # Provide a default value if 'subtype' is missing
        subtype = properties.get('subtype', 'unknown')
        uid = properties['uid']
        wkt_data = entry['wkt']

        class_label = f"{feature_type}-{subtype}"

        shape = wkt.loads(wkt_data)
        if shape.geom_type != "Polygon" or class_label not in class_label_to_int:
            continue

        # Extract the bounding box coordinates
        x_min = shape.bounds[0]
        y_min = shape.bounds[1]
        x_max = shape.bounds[2]
        y_max = shape.bounds[3]

        # Check if the bounding box is valid (width and height > 0)
        if x_min < x_max and y_min < y_max:
            # Extract the class label (e.g., 'building-no-damage')

            # Create a dictionary to store the annotation for this entry
            image_name = os.path.splitext(os.path.basename(json_file))[0]

            # Create a new entry for this image name if it does not exist
            if image_name not in annotations_dict:
                annotations_dict[image_name] = {
                    "boxes": [],
                    "labels": [],
                    "image_name": image_name
                }

            # Append the box and label to the existing entry for this image name
            annotations_dict[image_name]["boxes"].append(
                [x_min, y_min, x_max, y_max])
            annotations_dict[image_name]["labels"].append(
                class_label_to_int[class_label])

    return annotations_dict, image_name


def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, patience=25):
    best_val_loss = float('inf')
    best_model_state = None

    # Create a SummaryWriter for logging to TensorBoard
    writer = SummaryWriter()

    train_loader_len = len(train_loader)

    # Set the model to training mode
    model.train()

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        train_loss = 0.0  # Variable to store training loss for the epoch

        training_progress = Progress(f"Epoch {epoch + 1}/{num_epochs}")

        # Loop over the batches of images and targets in the train loader
        for i, (images, targets) in enumerate(train_loader):
            training_progress.print(i, train_loader_len)

            # Move the images and targets to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # Reset the gradients of the optimizer
            optimizer.zero_grad()

            try:
                # Forward pass the images and targets through the model and get the loss dictionary
                loss_dict = model(images, targets)
            except Exception as e:
                # starting with \n because we don't want to print it on top of the progress bar
                print("\nInvalid Box: ", str(e))
                continue

            # Sum up the losses from the loss dictionary
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()  # Accumulate the training loss for the epoch

            # Backward pass the losses and update the optimizer parameters
            losses.backward()
            optimizer.step()

        training_progress.stop()

        loss_progress = Progress(
            f"Calculating validation loss for epoch {epoch + 1}/{num_epochs}")
        # Compute average validation loss
        val_loss = 0.0
        with torch.no_grad():
            # Replace val_loader with your validation data loader
            for i, (val_images, val_targets) in enumerate(val_loader):
                loss_progress.print(i, len(val_loader))

                val_images = list(image.to(device) for image in val_images)
                val_targets = [{k: v.to(device) for k, v in t.items()}
                               for t in val_targets]
                val_loss_dict = model(val_images, val_targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_loss += val_losses.item()

        loss_progress.stop()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            num_epochs_since_improvement = 0
        else:
            num_epochs_since_improvement += 1
            if num_epochs_since_improvement >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        # Log the training and validation loss to TensorBoard
        writer.add_scalar('Loss/Train', losses.item(), epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Print the epoch number and completion message
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        print(f"Saving epoch {epoch + 1} model")

        # if the models directory does not exist, create it
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the trained model for later use
        torch.save(model.state_dict(),
                   os.path.join('models', f'epoch_{epoch + 1}.pth'))

    torch.save(best_model_state, 'object_detection_model.pth')
    writer.close()


def read_annotation(json_filename):
    image_name = os.path.splitext(os.path.basename(json_filename))[
        0]  # Get the image name without extension

    # Check if the JSON file should be processed based on the presence of 'wkt' key
    with open(json_filename, 'r') as f:
        data_parse = json.load(f)

        data_array = data_parse.get('features', {}).get('xy', [])

        for entry in data_array:
            if 'wkt' in entry:

                properties = entry['properties']
                feature_type = properties['feature_type']
                subtype = properties.get('subtype', 'unknown')
                class_label = f"{feature_type}-{subtype}"

                # Only process polygons with desired labels
                if class_label not in ["building-no-damage", "building-destroyed"]:
                    continue

                annotations = parse_json(json_filename, data_parse)
                return {image_name: annotations}

    return {}


def create_datasets(image_folder, json_files_folder, transform=None, test_size=0.2):
    # Get the list of all JSON files in the folder
    json_files = glob.glob(os.path.join(json_files_folder, '*.json'))

    # Initialize an empty dictionary to store annotations for each image
    all_annotations = {}

    with Pool(16) as p:  # Use 16 processes because why not
        progress = Progress("Creating datasets")
        progress.start()

        for i, data in enumerate(p.imap_unordered(read_annotation, json_files)):
            progress.print(i, len(json_files))

            all_annotations.update(data)

        progress.stop()

    print("Number of images with annotations:", len(all_annotations))

    # Split the annotations into training and validation sets
    train_annotations, val_annotations = train_test_split(
        list(all_annotations.values()), test_size=test_size)

    # Create the training and validation datasets
    train_dataset = CustomObjectDetectionDataset(
        image_folder, train_annotations, transform=transform)
    val_dataset = CustomObjectDetectionDataset(
        image_folder, val_annotations, transform=transform)

    return train_dataset, val_dataset


if __name__ == '__main__':
    device = torch.device(get_device())

    # Define paths to your dataset and JSON files folder
    image_folder = 'data/images'
    json_files_folder = 'data/labels'

    train_dataset, val_dataset = create_datasets(
        image_folder, json_files_folder, transform=transform)

    print("Number of training examples:", len(train_dataset))

    # Define data loader
    # data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Rest of the code for training the model (same as before)

    # Use a pre-trained Faster R-CNN model with ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)

    # Modify the model's output layer to match the number of target classes
    # class_labels = set(annotation['labels'] for annotation in annotations)
    num_classes = 7  # +1 for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    # Set device for training (GPU if available, else CPU)
    model = model.to(device)

    # Define the optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    # Train the model
    train_model(model=model, train_loader=train_loader,
                val_loader=val_loader, optimizer=optimizer, num_epochs=10)
