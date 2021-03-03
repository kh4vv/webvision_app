import glob
import os
import sys

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from inference import class_dict_extraction

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

'''
#quick_draw_class_map = {0: 'baseball', 1: 'birthday cake', 2: 'broccoli', 
    3: 'animal migration', 4: 'aircraft carrier', 5: 'bat', 
    6: 'binoculars', 7: 'bulldozer', 8: 'boomerang', 9: 'bee', 
    10: 'anvil', 11: 'bear', 12: 'airplane', 13: 'bench', 
    14: 'bird', 15: 'basket', 16: 'bicycle', 17: 'angel', 
    18: 'bucket', 19: 'bridge', 20: 'belt', 21: 'barn', 
    22: 'bread', 23: 'axe', 24: 'book', 25: 'backpack', 
    26: 'bed', 27: 'banana', 28: 'beard', 29: 'beach', 
    30: 'blackberry', 31: 'blueberry', 32: 'basketball', 
    33: 'bottlecap', 34: 'bathtub', 35: 'bowtie', 
    36: 'broom', 37: 'ant', 38: 'The Great Wall of China', 
    39: 'bandage', 40: 'The Eiffel Tower', 41: 'apple', 
    42: 'ambulance', 43: 'baseball bat', 44: 'bracelet', 
    45: 'asparagus', 46: 'alarm clock', 47: 'The Mona Lisa', 
    48: 'arm', 49: 'brain'}
'''
#Make dictionary for quick draw class map with file path
path = '/home/ubuntu/hdd_ext/hdd4000/quickdraw_dataset'
quick_draw_class_map = class_dict_extraction(path, 'npy')
#Quick draw class category
cat1 = [9, 23, 26,29,34,52,57,67,84,94,95,98,102,106,119,120,128,132,144,150,162,177,179,186,189,191,194,204,207,211,221,225,238,239,260,261,271,272,279,284,307,312,336,343] #Living creature (animal + insects)

class MnistDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.image_ids = glob.glob(path + '/**/**/*')
        self.labels = [int(data.split('/')[4]) for data in self.image_ids]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_ids[idx])
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class QuickDrawDataset(Dataset):
    def __init__(self, path, cat=cat1, transforms=None):
        self.files = sorted(glob.glob(path + '/*'))
        self.all_x =[]
        self.all_y =[]
        self.cat = cat

        self.image_ids, self.label_ids = self.load_data()
        self.transform = transform

        self.image_ids = np.vstack(self.image_ids)
        self.label_ids = np.vstack(self.label_ids)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image = self.image_ids[idx]
        label_id = int(self.label_ids[idx])

        if self.transform is not None:
            image = self.transform(image)
        return image, label_id
    
    def load_data(self):
        """
        #define dictionary type datastructure
        images_dict = {}
        for file in self.files:
            images = np.load(file)
            images = images.astype('float32') / 255.
            images = images[0:num_data, :] # Subset only 15000 data(There are too many!!)
            images = images.flatten() # Make it to 1D array
            #extract labels
            labels = os.path.splitext(os.path.basename(file).split('_')[-1])[0]
            #find label id (key) from the labels (values) in dictionary
            label_ids = list(quick_draw_class_map.keys())[list(quick_draw_class_map.values()).index(labels)]
            images_dict[label_ids]=images
            print(file)

        label_ids = list(images_dict.keys())
        images = list(images_dict.values())
        #We need to pull the image one by one
        #Label ID
        label_ids_total=[]
        for ids in label_ids:
            for _ in range(num_data):
                label_ids_total.append(ids)
        #Image
        img_reshape = [x.reshape(num_data,-1) for x in images]
        print(np.array(img_reshape).shape)
        images_total = [elem for twod in img_reshape for elem in twod]
        print(np.array(images_total).shape)
        print(np.array(label_ids_total).shape)
        
        return images_total , label_ids_total
        """
        count = 0
        
        all_x = []
        ids = 0
        for file in self.files:
            if count in self.cat:
                images = np.load(file)
                images = images.astype('float32') / 255.
                images = images[0:15000, :] # Subset only 15000 data(There are too many!!)
                images = images.reshape(-1, 28, 28)
                all_x.append(images.copy())
                
                label_ids = [ids for _ in range(len(images))]
                label_ids = np.array(label_ids).astype('float32')
                label_ids = label_ids.reshape(label_ids.shape[0], 1)
                self.all_y.append(label_ids)
                labels = os.path.splitext(os.path.basename(file).split('_')[-1])[0]
                ids += 1
                print(count)
            count += 1
            
        return all_x, self.all_y

class LandmarkDataset(Dataset):
    def __init__(self, transforms=None):
        self.image_ids = glob.glob('./data/train/**/**/*')
        with open('./data/train.csv') as f:
            labels = list(csv.reader(f))[1:]
            self.labels = {label[0]: int(label[1]) for label in labels}
            print(self.labels)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
        label = self.labels[label]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image, label
