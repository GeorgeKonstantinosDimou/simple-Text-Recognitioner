from PIL import Image
from torch.utils.data import Dataset
import tqdm
import torch
import os
import scipy.io as sio
import random


def tol(l):
    l = l.lower()
    if ord(l) < ord('a'):
        return ord(l) - ord('0') + 1
    else:
        return ord(l) - ord('a') + 11
          
def get_dataset(dataname):
    dataset = {
        'IIIT5K_train': ['/home/dimou/Files/textRec/datasets/IIIT5k/', '/home/dimou/Files/textRec/datasets/IIIT5k/trainCharBound.mat'],
        'IIIT5K_test': ['/home/dimou/Files/textRec/datasets/IIIT5k/', '/home/dimou/Files/textRec/datasets/IIIT5k/testCharBound.mat'],
    }
    return dataset[dataname]

class MyDataset(Dataset):

    def __init__(self, datanames, transform=None, target_transform=None):
        if not isinstance(datanames, list):
            datanames = [datanames]
        imgs = []
        for dataname in datanames:
            im_dir, txt_path = get_dataset(dataname)
            imgs.extend(self.get_data(im_dir, txt_path))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        try:
            img = Image.open(fn).convert('RGB')
        except IOError:
            print('Corrupted image for %s' % fn)
            return self[index + 1]
        if img.height > img.width:
            if random.random() > 0.5:
                img = img.transpose(Image.ROTATE_90)
            else:
                img = img.transpose(Image.ROTATE_270)
        #lat = torch.zeros(30)
        lat = torch.full(size = (30, ), fill_value = 38)
        for i in range(len(label)):
            lat[i] = int(label[i])
            
        """This peace of code exists for the case of consecutively same characters, i.g hello
            so it needs to input the "blank" character in the targets. In case of Pytorch this
            is not needed as blank character cannot exist in the targets"""
        # i = 0
        # while i < len(label):
        #     lat[i] = int(label[i])
        #     i += 1
        #     if(lat[i-1]) == int(label[i]):
        #         lat[i] = 0
        #         lat[i+1] = int(label[i])
        #         i += 2
        #print(lat)
        
        length = len(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, lat, length

    def __len__(self):
        return len(self.imgs)
    
    def get_data(self, im_dir, txt_path=None):
        if txt_path:
            if 'mat' in txt_path:
                data = sio.loadmat(txt_path)
                da = data[txt_path.split('/')[-1][:-4]][0]
                imgs = []
                for gt in tqdm.tqdm(da, ascii=True):
                    imn = im_dir + gt[0][0]
                    #print(imn)
                    la = gt[1][0].strip()
                    #print(la)
                    if len(la) > 30:
                        continue
                    la = [tol(l) for l in la]
                    
                    #la.insert(0, 37)
                    la.append(37)
                    #print(la)
                    
                    imgs.append([imn, la])
            else:
                with open(txt_path) as f:
                    gts = f.readlines()
                imgs = []
                for gt in tqdm.tqdm(gts, ascii=True):
                    imn = im_dir + gt.strip().split(' ')[0]
                    la = gt.strip().split(' ')[1:]
                    if len(la) > 30:
                        continue
                    imgs.append([imn, la])
        else:
            imgs = []
            ims = os.listdir(im_dir)
            for im in ims:
                imgs.append([os.path.join(im_dir, im), [0]])
        return imgs
    
    def get_mean_std(self):
        mean, std = 0, 0
        
        for i in range(self.__len__()):
            img, _, _ = self.__getitem__(i)
            mean += img.mean(dim = [1, 2])
            std += img.std(dim = [1, 2])
        
        mean.tolist()
        std.tolist()
        mean /= self.__len__()
        std /= self.__len__()
        
        return mean, std