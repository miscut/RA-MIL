import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def read_bag(train_list, x, data_path):
    patient_list = []
    label_list = []
    for t in train_list:  # patient level,eg:['3', 0],<class 'str'>
        patient = int((t.split(',')[0].replace("['", '')).replace("'", '')) + x # 患者编号，如：3
        patient_path = os.path.join(data_path, str(patient))
        label = t.split(',')[1].replace("]", '') # 患者标签，如：0
        patient_list.append(patient_path)
        label_list.append(label)
    return patient_list,label_list # 返回患者路径和标签

class aaMILDataset(Dataset):
    def __init__(self, df, transform=None):
        super(aaMILDataset, self).__init__() # 对继承自父类的属性进行初始化
        self.df = df
        self.transform = transform # 默认不数据转换，随用随叫

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.df['label']

    def __getitem__(self, idx):
        row = self.df.iloc[idx] # 根据索引
        bag_path = row['bag']
        file_list = os.listdir(bag_path)

        pixel_list = []
        file_list.sort(key=lambda x: int((x.split('.')[0]).split('_')[1]))  # 文件排序
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
            if im_size / pixelest >= 0: # mp写不进类里，只能手动改了
                instance_list.append(f)

        for i in instance_list:
            instance_path = os.path.join(bag_path, i)
            img = Image.open(instance_path)
            img = self.transform(img)
            img = torch.unsqueeze(img, dim=0) # 增加一个维度，(3,64,64)变为(1,3,64,64)
            if instance_list.index(i) == 0:
                bag = img  # 第一个tensor不需要拼接
            else:
                bag = torch.cat((bag, img), 0)

        label = row['label']
        label = np.array(int(label)) # 不能是浮点
        label = torch.from_numpy(label) # 变成tensor
        label = label.long() # 变成long

        return bag_path, bag, label # 输出患者为了测试中的后续分析，模型仅需要后两个输出