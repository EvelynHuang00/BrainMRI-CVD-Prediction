import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import nilearn
import nibabel as nib
import os

class PatientDataset(Dataset):
    def __init__(self, valid_id_dir, image_base_path, transform=None):
        self.id = pd.read_csv(valid_id_dir)
        self.image_base_path = image_base_path
        self.min_coords = [56, 61, 34]
        self.max_coords = [200, 220, 217]
        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        eid = str(self.id.iloc[idx, 0])  # Get the patient ID
        img_path = os.path.join(self.image_base_path, f"{eid}_20263_2_0/mri/brainmask.mgz")
        # Load the .mgz file
        img = nib.load(img_path)
        image_data = img.get_fdata()
        
        image_data = image_data[self.min_coords[0]:self.max_coords[0]+1,
                                self.min_coords[1]:self.max_coords[1]+1,
                                self.min_coords[2]:self.max_coords[2]+1]
        image_data = np.float32(image_data)
        image_data = torch.from_numpy(image_data)
        # insert dimension for channel
        image_data = image_data.unsqueeze(0)

        # Convert the first slice of the .mgz file to a PIL image
        # image = Image.fromarray(image_data[:, :, 0]).convert("RGB")
        label = self.id.iloc[[idx], :]['I67']
        label = np.float32(label)
        label = torch.from_numpy(label)
        #if self.transform:
            #image = self.transform(image)
        return image_data, label
