import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import nilearn
import nibabel as nib
import numpy as np
import os

class PatientDataset(Dataset):
    def __init__(self, valid_id_dir, image_base_path, transform=None):
        self.id = pd.read_csv(valid_id_dir)
        self.image_base_path = image_base_path
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        eid = str(self.id.iloc[idx, 0])  # Get the patient ID
        img_path = os.path.join(self.image_base_path, f"{eid}_20252_2_0/T1/T1_brain_to_MNI.nii.gz")
        # Load the .mgz file
        img = nib.load(img_path) # convert this into a torch tensor 
        image_data = img.get_fdata()
        # Calculate non-zero coordinates
        non_zero_coords = np.argwhere(image_data > 0)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0)
        
        # Crop the image using min and max coordinates
        image_data = image_data[min_coords[0]:max_coords[0]+1,
                                min_coords[1]:max_coords[1]+1,
                                min_coords[2]:max_coords[2]+1]
        
        # Decrease precision to make images use less memory
        image_data = np.float32(image_data)
        
        # Create a mask for non-zero elements
        image_data_mask = image_data != 0
        
        # Normalize non-zero elements to z-scores
        non_zero_mean = image_data[image_data_mask].mean()
        non_zero_std = image_data[image_data_mask].std()
        image_data[image_data_mask] = (image_data[image_data_mask] - non_zero_mean) / non_zero_std
        
        
        # Convert the image data to a torch tensor
        image_data = torch.from_numpy(image_data)
        
        # Insert dimension for channel
        image_data = image_data.unsqueeze(0)
        
        # Convert the first slice of the .mgz file to a PIL image
        # image = Image.fromarray(image_data[:, :, 0]).convert("RGB")
        label = self.id.iloc[[idx], :]['I67']
        label = np.float32(label)
        label = torch.from_numpy(label)
        #if self.transform:
            #image = self.transform(image)哦
        return image_data, label
