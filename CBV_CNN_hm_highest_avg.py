import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from PIL import Image
import matplotlib.pyplot as plt
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
input_size = config['model']['input_size']
num_classes = config['model']['num_classes']
input_channels = config['model']['input_dimensions']['input_channels']
depth = config['model']['input_dimensions']['depth']
height = config['model']['input_dimensions']['height']
width = config['model']['input_dimensions']['width']
num_epochs = config['training_parameters']['epochs']
save_model_path = config['training_parameters']['pretrain']


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

# Function to compute saliency
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
save_dir_hm = './heatmap_highest_0.003'

# Iterate over all batches in the testing dataset
for batch_idx, (images, labels) in enumerate(test_loader):
    # Initialize an array to accumulate saliency maps for averaging
    accumulated_saliency = None

    for i in range(images.size(0)):
        img = images[i].cpu().numpy()

        # Compute saliency map and input image
        slc, input_img = saliency(img, net)

        # Convert input and saliency map to NumPy arrays
        input_np = input_img.numpy()
        saliency_np = slc.numpy()

        # Accumulate saliency maps
        if accumulated_saliency is None:
            accumulated_saliency = saliency_np
        else:
            accumulated_saliency += saliency_np

    # Compute average saliency map for the current patient
    average_saliency = accumulated_saliency / images.size(0)

    # Find the coordinates of the maximum point in the average saliency map
    max_index = np.unravel_index(np.argmax(average_saliency, axis=None), average_saliency.shape)
    print("Max average saliency coordinate:", max_index)

    # Prepare for saving images for the point with maximum average saliency
    directions = ['Sagittal', 'Axial', 'Coronal']
    slices = [
        (max_index[0], slice(None), slice(None)),  # Sagittal slice
        (slice(None), max_index[1], slice(None)),  # Axial slice
        (slice(None), slice(None), max_index[2])   # Coronal slice
    ]

    for direction, slice_index in zip(directions, slices):
        # Extract the original and saliency slices
        original_slice = input_np[0, slice_index[0], slice_index[1], slice_index[2]]  # Extract 2D slice
        saliency_slice = average_saliency[slice_index[0], slice_index[1], slice_index[2]]  # Extract 2D slice

        # Apply mask to the saliency slice
        mask = original_slice > 0
        masked_saliency = saliency_slice * mask

        # Normalize the masked saliency map
        masked_saliency = (masked_saliency - np.min(masked_saliency)) / (np.max(masked_saliency) - np.min(masked_saliency) + 1e-8)

        # Save the masked saliency map
        saliency_map_path = f'{save_dir_hm}/average_saliency_map_patient{batch_idx}_{direction}.png'
        plt.figure(figsize=(5, 5))
        plt.imshow(masked_saliency, cmap='hot')
        plt.colorbar()
        plt.title(f'{direction} Masked Average Saliency')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(saliency_map_path)
        plt.close()

        # Display and save the original slice with masked saliency overlay
        overlay_path = f'{save_dir_hm}/average_saliency_overlay_patient{batch_idx}_{direction}.png'
        plt.figure(figsize=(5, 5))
        plt.imshow(original_slice, cmap='gray')
        plt.imshow(masked_saliency, cmap=plt.cm.hot, alpha=0.7)  # Adjust alpha for transparency
        plt.title(f'{direction} Overlay')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(overlay_path)
        plt.close()
        
# Iterate over all batches in the testing dataset
# for batch_idx, (images, labels) in enumerate(test_loader):
#     for i in range(images.size(0)):
#         img = images[i].cpu().numpy()
        
#         # Generate saliency map
#         slc, input_img = saliency(img, net)
        
#         plt.figure(figsize=(10, 10))
        
#         # Plot the original slice
#         plt.subplot(1, 2, 1)
#         original_slice = input_img[0, input_img.shape[1] // 2, :, :].numpy()
#         plt.imshow(original_slice, cmap='gray')
#         plt.xticks([])
#         plt.yticks([])
#         plt.title("Original Slice")
        
#         # Plot the saliency map
#         plt.subplot(1, 2, 2)
#         saliency_slice = slc[input_img.shape[1] // 2, :, :].numpy()
#         mask = original_slice > 0
#         saliency_slice *= mask
        
#         # Normalize the saliency_slice
#         saliency_slice = (saliency_slice - np.min(saliency_slice)) / (np.max(saliency_slice) - np.min(saliency_slice))
        
#         plt.imshow(saliency_slice, cmap=plt.cm.hot, alpha=0.7)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title("Saliency Map with Mask")
        
#         heatmap_path = f'{save_dir_hm}/saliency_batch{batch_idx}_image{i}_mask.png'
#         plt.savefig(heatmap_path)
#         plt.close()

# heat map for one patient only
# for batch_idx, (images, labels) in enumerate(test_loader):
#     for i in range(images.size(0)):
#         img = images[i].cpu().numpy()
        
#         # Generate saliency map
#         slc, input_img = saliency(img, net)
        
#         # Convert tensors to numpy arrays
#         input_img = input_img.numpy()
#         slc = slc.numpy()
        
#         # Get the number of slices in the depth dimension
#         num_slices = input_img.shape[1]
#         print(f"Number of slices: {num_slices}")
        
#         # Iterate over all slices in the depth dimension
#         for slice_idx in range(num_slices):
#             plt.figure(figsize=(20, 10))
            
#             # Plot the original slice
#             plt.subplot(1, 2, 1)
#             original_slice = input_img[0, slice_idx, :, :]
#             plt.imshow(original_slice, cmap='gray')
#             plt.xticks([])
#             plt.yticks([])
#             plt.title(f"Original Slice {slice_idx}")
            
#             # Plot the saliency map
#             plt.subplot(1, 2, 2)
#             saliency_slice = slc[0, slice_idx, :, :]
#             mask = original_slice > 0
#             saliency_slice *= mask
            
#             # Normalize the saliency_slice
#             saliency_slice = (saliency_slice - np.min(saliency_slice)) / (np.max(saliency_slice) - np.min(saliency_slice))
            
#             plt.imshow(saliency_slice, cmap=plt.cm.hot, alpha=0.7)
#             plt.xticks([])
#             plt.yticks([])
#             plt.title(f"Saliency Map with Mask {slice_idx}")
            
#             heatmap_path = f'{save_dir_hm}/saliency_batch{batch_idx}_image{i}_slice{slice_idx}_mask.png'
#             plt.savefig(heatmap_path)
#             plt.close()
        
print("--- %s seconds elapsed ---" % (time.time() - start_time))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)