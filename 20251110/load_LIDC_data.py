import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            img = value['image']  # np.ndarray (H,W)
            m = value['masks']  # list 长度4，每个是 (H,W) 的 np.ndarray

            # 统一到 float32，值域裁剪到 [0,1]
            img = np.asarray(img, dtype=np.float32)
            # 把 list -> (4,H,W)
            m = np.stack([np.asarray(t) for t in m], axis=0).astype(np.float32)

            img = np.clip(img, 0.0, 1.0)
            m = np.clip(m, 0.0, 1.0)

            self.images.append(img)  # (H,W)
            self.labels.append(m)  # (4,H,W)
            self.series_uid.append(value.get('series_uid', str(key)))

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)