# this file is for creating heatmaps in coronal, sagittal and axial views
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import yaml
import time
import resource
import time
from data.dataset import PatientDataset
import torch.optim as optim
import cv2, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve

start_time = time.time()

# Load configuration from file
config_path = os.path.join(".", "config_matched.yml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)
print('Finish loading the config')
# Print to verify
# print(config)

# Accessing configuration parameters
image_base_path = config['data']['image_base_path']
train_csv = config['data']['train_data_path']
val_csv = config['data']['val_data_path']
test_csv = config['data']['test_data_path']
batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']

# Create datasets
train_dataset = PatientDataset(valid_id_dir=train_csv, image_base_path=image_base_path, transform=None)
val_dataset = PatientDataset(valid_id_dir=val_csv, image_base_path=image_base_path, transform=None)
test_dataset = PatientDataset(valid_id_dir=test_csv, image_base_path=image_base_path, transform=None)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('Finish creating data loaders')

class ComplexCNN3D(nn.Module):
    def __init__(self):
        super(ComplexCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool3d(3, stride=2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool3d(3, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool3d(3, stride=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=5, padding=2)
        self.pool4 = nn.MaxPool3d(3, stride=2)

        self.conv5 = nn.Conv3d(256, 512, kernel_size=5, padding=2)
        self.pool5 = nn.MaxPool3d(3, stride=2)

        # Increasing the number of neurons in the fully connected layers
        self.fc1 = nn.Linear(175616, 8192)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(8192, 2048)  # Increased number of neurons
        self.fc3 = nn.Linear(2048, 512)  # Increased number of neurons
        self.fc4 = nn.Linear(512, 1)  # Output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# before creating, load a model here
net = ComplexCNN3D()
checkpoint = torch.load('/external/rprshnas01/kcni/evhuang/pbsweb/cerebrovascular/checkpoints/final_2310664.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
net.eval()
print('Model loaded successfully')

# Define transformation classes
class ToTensor3D:
    def __call__(self, sample):
        image = sample['image']
        return {'image': torch.from_numpy(image).float()}

# image normalization is done before, so mean is 0 and std is 1 here
class Normalize3D:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image = (image - self.mean) / self.std
        return {'image': image}

class InvNormalize3D:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        image = tensor.clone()
        image = image * self.std + self.mean
        return image

# Define the transformation and inverse transformation
transform = transforms.Compose([
    ToTensor3D(),
    Normalize3D(mean=0.0, std=1.0)
])

inv_normalize = InvNormalize3D(mean=0.0, std=1.0)

def saliency(img, model):
    # Disable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    sample = {'image': img}
    transformed_sample = transform(sample)
    input = transformed_sample['image']
    input = input.unsqueeze(0)  # Add batch dimension

    input.requires_grad = True

    preds = model(input)
    
    # Since the output is a single value, use preds directly
    score = preds
    
    score.backward()

    # Compute the maximum absolute gradient across all channels
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)

    # Normalize the saliency map
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    with torch.no_grad():
        # Apply inverse normalization to the input image
        input_img = inv_normalize(input[0])

    # Return the saliency map and input image for later saving
    return slc, input_img

# Directory to save the heatmaps
save_dir_hm = './heatmap_xyz'


# create saliency map and then overlay in sagittal view 
# Iterate over all batches in the testing dataset
# for batch_idx, (images, labels) in enumerate(test_loader):
#     for i in range(images.size(0)):
#         img = images[i].cpu().numpy()
        
#         slc, input_img = saliency(img, net)
        
#         plt.figure(figsize=(10, 10))
#         plt.subplot(1, 2, 1)
#         # Select a 2D slice from the 3D image
#         original_slice = input_img.numpy()[0, input_img.shape[1] // 2, :, :]
#         plt.imshow(original_slice, cmap='gray')
#         plt.xticks([])
#         plt.yticks([])
        
#         plt.subplot(1, 2, 2)
#         # Apply mask to the saliency slice
#         saliency_slice = slc.numpy()[input_img.shape[1] // 2, :, :]
#         mask = original_slice > 0
#         saliency_slice *= mask
        
#         # Enhance contrast of saliency slice
#         saliency_slice = (saliency_slice - np.min(saliency_slice)) / (np.max(saliency_slice) - np.min(saliency_slice))
        
#         # Display the original slice with saliency map overlay
#         plt.imshow(original_slice, cmap='gray')
#         plt.imshow(saliency_slice, cmap=plt.cm.hot, alpha=0.7)  # Adjust alpha for transparency
#         plt.xticks([])
#         plt.yticks([])

#         heatmap_path = f'{save_dir_hm}/saliency_overlay_batch{batch_idx}_image{i}.png'
#         plt.savefig(heatmap_path)
#         plt.close()


# Function to generate and save heatmaps
def generate_heatmaps(img, slc, axis, slice_idx, save_dir, batch_idx, image_idx, axis_label):
    plt.figure(figsize=(10, 10))
    
    # Select the appropriate slice based on the axis
    if axis == 'x':
        original_slice = img[slice_idx, :, :]
        saliency_slice = slc[slice_idx, :, :]
    elif axis == 'y':
        original_slice = img[:, slice_idx, :]
        saliency_slice = slc[:, slice_idx, :]
    elif axis == 'z':
        original_slice = img[:, :, slice_idx]
        saliency_slice = slc[:, :, slice_idx]
    
    # Convert to NumPy array if necessary
    if not isinstance(original_slice, np.ndarray):
        original_slice = np.array(original_slice)
    if not isinstance(saliency_slice, np.ndarray):
        saliency_slice = np.array(saliency_slice)
    
    # Apply mask to the saliency slice
    mask = original_slice > 0
    saliency_slice *= mask
    
    # Normalize the saliency_slice
    saliency_min = np.min(saliency_slice)
    saliency_max = np.max(saliency_slice)
    if saliency_max - saliency_min != 0:  # Avoid division by zero
        saliency_slice = (saliency_slice - saliency_min) / (saliency_max - saliency_min)
    
    # Plot the original slice
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Original Slice ({axis_label}-axis)")
    
    # Plot the saliency map with overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_slice, cmap='gray')
    plt.imshow(saliency_slice, cmap=plt.cm.hot, alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Saliency Map ({axis_label}-axis)")
    
    # Save the result
    heatmap_path = os.path.join(save_dir, f'saliency_overlay_batch{batch_idx}_image{image_idx}_{axis_label}axis_slice{slice_idx}.png')
    plt.savefig(heatmap_path)
    plt.close()

# Iterate over all batches in the testing dataset
for batch_idx, (images, labels) in enumerate(test_loader):
    for i in range(images.size(0)):
        img = images[i].cpu().numpy()
        
        # Generate saliency map
        slc, input_img = saliency(img, net)
        
        # Convert tensors to numpy arrays and squeeze
        input_img = input_img.squeeze()  # Ensure it has shape [D, H, W]
        slc = slc.squeeze()  # Ensure it has shape [D, H, W]
        
        # Get the middle slice indices for each axis
        x_middle_slice_idx = input_img.shape[0] // 2  # Middle slice along x-axis
        y_middle_slice_idx = input_img.shape[1] // 2  # Middle slice along y-axis
        z_middle_slice_idx = input_img.shape[2] // 2  # Middle slice along z-axis
        
        # Generate heatmaps along x, y, and z axes
        generate_heatmaps(input_img, slc, 'x', x_middle_slice_idx, save_dir_hm, batch_idx, i, 'x')
        generate_heatmaps(input_img, slc, 'y', y_middle_slice_idx, save_dir_hm, batch_idx, i, 'y')
        generate_heatmaps(input_img, slc, 'z', z_middle_slice_idx, save_dir_hm, batch_idx, i, 'z')
        
print("--- %s seconds elapsed ---" % (time.time() - start_time))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)