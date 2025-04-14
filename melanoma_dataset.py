
import os
from torch.utils.data import Dataset
from PIL import Image

class MelanomaDataset(Dataset):
    """Custom dataset class using Pytorch Dataset."""
    
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'melanoma_cancer_dataset/train' or 'melanoma_cancer_dataset/test'
        """
        
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Get all image paths and labels
        for label_name in ['benign', 'malignant']:
            label_dir = os.path.join(root_dir, label_name)
            label = 0 if label_name == 'benign' else 1
            for filename in os.listdir(label_dir):
                if filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):

        # Get corresponding image path and label
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label