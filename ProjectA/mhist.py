import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MHISTDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, train=True):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if train:
            self.df = self.df[self.df['Partition'] == 'train']
        else:
            self.df = self.df[self.df['Partition'] != 'train']

        self.image_names = self.df['Image Name'].values
        self.labels = self.df['Majority Vote Label'].values

        self.cat_to_num = {'HP': 0, 'SSA': 1}
        self.classes = ["hyperplastic polyp", "sessile serrated adenoma"]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_dir / self.image_names[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        label = self.cat_to_num[self.labels[idx]]
        
        return image, label


def get_MHIST_dataset(data_path, im_size=(224, 224)):
    channel = 3
    num_classes = 2
    mean = [0.7378, 0.6486, 0.7752]
    std = [0.1972, 0.2437, 0.1703]
    csv_file = Path(data_path) / 'annotations.csv'
    image_dir = Path(data_path) / 'images/'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize(im_size)])
    dst_train = MHISTDataset(csv_file, image_dir, transform, train=True)
    dst_test = MHISTDataset(csv_file, image_dir, transform, train=False)
    class_names = dst_train.classes

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
