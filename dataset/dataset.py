import numpy as np

from os import listdir
from PIL import Image

from torch.utils.data import Dataset

class ToothDataset(Dataset):
    def __init__(self, root='DOF_depthmap_256/', is_train=True):
        super().__init__()
        self.root = root
        self.is_train = is_train
        if not is_train:
            self.root_list = ['Test_data/input_Finger_test/','Test_data/input_Gro_test/','Test_data/input_Gap_test/','Test_data/Opp_test/','Test_data/input_test/','Test_data/Obj_test/']
        else:
            self.root_list = ['Train_data/input_Finger_Mask/','Train_data/input_Gro_Mask/','Train_data/input_Gap_Mask/','Train_data/input_Opp/','Train_data/input_Mask/','Train_data/Obj/']     

        self.input_list = []
        for path in self.root_list:
            self.input_list.append([x for x in listdir(self.root+path)])

    def __getitem__(self, index):
        img_list = []
        for files, file_root in zip(self.input_list[:-1], self.root_list[:-1]):
            img = Image.open(self.root+file_root+files[index]).convert('L')
            img = np.asarray(img)
            img = img/255
            img = img.astype(np.float32)
            img_list.append(img)
        img_list = np.array(img_list, dtype=np.float32)

        label = Image.open(self.root+self.root_list[-1]+self.input_list[-1][index]).convert('L')
        label = np.asarray(label)
        label = label/255
        label = label.astype(np.float32)
        label = np.array([label], dtype=np.float32)

        return img_list, label, self.input_list[-1][index]
    
    def __len__(self):
        return len(self.input_list[0])

class ToothDataset_3Ch(Dataset):
    def __init__(self, root='DOF_depthmap_256/', is_train=True):
        super().__init__()
        self.root = root
        self.is_train = is_train
        if not is_train:
            self.root_list = ['Test_data/input_Gap_test/','Test_data/Opp_test/','Test_data/input_test/','Test_data/Obj_test/']
        else:
            self.root_list = ['Train_data/input_Gap_Mask/','Train_data/input_Opp/','Train_data/input_Mask/','Train_data/Obj/']     

        self.input_list = []
        for path in self.root_list:
            self.input_list.append([x for x in listdir(self.root+path)])

    def __getitem__(self, index):
        img_list = []
        for files, file_root in zip(self.input_list[:-1], self.root_list[:-1]):
            img = Image.open(self.root+file_root+files[index]).convert('L')
            img = np.asarray(img)
            img = img/255
            img = img.astype(np.float32)
            img_list.append(img)
        img_list = np.array(img_list, dtype=np.float32)

        label = Image.open(self.root+self.root_list[-1]+self.input_list[-1][index])
        label = np.asarray(label)
        label = label.transpose((2, 0, 1))
        label = label/255
        label = label.astype(np.float32)

        return img_list, label, self.input_list[-1][index]
    
    def __len__(self):
        return len(self.input_list[0])

if __name__ == '__main__':
    test = ToothDataset(is_train=True)
    image, label, _ = test[0]