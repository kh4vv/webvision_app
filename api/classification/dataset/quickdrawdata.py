from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



classname = ['bear', 'dog', 'bat'] #we should add more class in future

class Quickdraw(dataset):
    def __init__(self, classname = None, datasize, transforms = None):
        self.datasize = datasize
        self.qd = QuickDrawData(max_drawings=self.datasize)

        if classname is None:
            self.classname = qd.drawing_names
        else:
            self.classname = classname
        self.transfroms = transforms
       
        self.label_ids = getLabelID(self.classname, self.datasize)
        self.img_ids = getImageID(self.classname, self.datasize)

    def getLabelID(classname, datasize):
        label_ids = []
        for i in range(len(classname)):
            label_id = [i for _ in range(datasize)]
            label_ids.append(label_id)
        label_ids = [element for sublist in label_ids for element in sublist]
        return label_ids

    def getImageID(classname, datasize):
        img_ids = []
        for i in range(len(classname)):
            for j in range(datasize):
                img = self.qd.get_drawing(classname[i], index = j)
                img_ids.append(img.image)
        return img_ids
    
    def __getitem__(self, idx):
        return self.img_ids[idx], self.label_ids[idx]

    def __len__(self):
        return self.datasize * len(self.classname)

