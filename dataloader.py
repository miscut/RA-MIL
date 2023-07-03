import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def label_separate(training_list):
    list_0 = []
    list_1 = []
    for tl in training_list:
        tl_label = tl.split(',')[1].replace("]", '')
        if int(tl_label) == 0:
            list_0.append(tl)
        else:
            list_1.append(tl)
    return list_0,list_1

def read_bag(train_list, x, data_path):
    patient_list = []
    label_list = []
    for t in train_list:  # patient level,eg:['3', 0],<class 'str'>
        patient = int((t.split(',')[0].replace("['", '')).replace("'", '')) + x # patient num
        patient_path = os.path.join(data_path, str(patient))
        label = t.split(',')[1].replace("]", '')
        patient_list.append(patient_path)
        label_list.append(label)
    return patient_list,label_list # patient path and label

class aaMILDataset(Dataset):
    def __init__(self, df, transform=None):
        super(aaMILDataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.df['label']

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bag_path = row['bag']
        file_list = os.listdir(bag_path)

        pixel_list = []
        file_list.sort(key=lambda x: int((x.split('.')[0]).split('_')[1]))
        for f in file_list:
            file_path = os.path.join(bag_path, f)
            im = cv2.imread(file_path)
            pixel_list.append(im.shape[0] * im.shape[1])
        pixelest = max(pixel_list)

        instance_list = []
        for f in file_list:
            file_path = os.path.join(bag_path, f)
            im = cv2.imread(file_path)
            im_size = im.shape[0] * im.shape[1]
            if im_size / pixelest >= 0:
                instance_list.append(f)

        for i in instance_list:
            instance_path = os.path.join(bag_path, i)
            img = Image.open(instance_path)
            img = self.transform(img)
            img = torch.unsqueeze(img, dim=0)
            if instance_list.index(i) == 0:
                bag = img
            else:
                bag = torch.cat((bag, img), 0)

        label = row['label']
        label = np.array(int(label))
        label = torch.from_numpy(label)
        label = label.long()

        return bag_path, bag, label