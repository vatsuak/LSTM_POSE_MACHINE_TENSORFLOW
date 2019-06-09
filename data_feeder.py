#feeder block for the torch model 

import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import scipy.misc
from PIL import Image


class Feeder():

    def __init__(self, data_dir, label_dir, train, temporal, joints, sigma):
        self.height = 368
        self.width = 368

        self.data_dir = data_dir   # eg './datadir/' should contain all the images in correct order of their timestamp eg. L####.jpg
        self.label_dir = label_dir #should contain the JSON file where the labels are stored eg. ($PATH$)/001L0.json

        self.temporal = temporal
        # self.transform = transform
        self.joints = joints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.temporal_dir = []

        self.train = train
        if self.train is True:
            self.gen_temporal_dir(1)
        else:
            self.gen_temporal_dir(temporal)

    def normalize(self, pic):
        """
        convert each image to float32 and normalize them
        :param im: 368*368*3 uint8-type      range: 0-255
        :return:   368*368*3 float32-type    range: 0-1 
        """
        if pic.mode == 'I':
            img = np.array(pic, np.int32, copy=False)
        elif pic.mode == 'I;16':
            img = np.array(pic, np.int16, copy=False)
        else:
            img = np.array(pic)
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.reshape(pic.size[0], pic.size[1], nchannel)
        return img.astype(np.float32)/255.0



    def gen_temporal_dir(self, step):
        """
        build temporal directory in order to guarantee get all images has equal chance to be trained
        for train dataset, make each image has the same possibility to be trained

        :param step: for training set, step = 1, for test set, step = temporal
        :return:
        """
        # the temporal size should be atleast as big as the number of images
        imgs = os.listdir(self.data_dir)  # ['L0005.jpg', 'L0011.jpg', ......]
        imgs.sort()

        img_num = len(imgs)
        print("--------------------Preparing Data---------------------------------------")
        for i in tqdm(range(0, img_num - self.temporal + 1, step)):
            tmp = []
            for k in range(i, i + self.temporal):
                tmp.append(os.path.join(self.data_dir, imgs[k]))
            self.temporal_dir.append(tmp)  # temporal_dir = [['datadir/L0005.jpg', 'datadir/L0011.jpg', ......],['datadir/L0005.jpg', 'datadir/L0011.jpg', ......]]

        self.temporal_dir.sort()
        print('total numbers of image sequence is ' + str(len(self.temporal_dir)))

    # def __len__(self):
    #     if self.train is True:
    #         length = len(self.temporal_dir)//self.temporal
    #         # print(self.temporal_dir)
    #     else:
    #         length = len(self.temporal_dir)
    #     return length

    def __getitem__(self, idx):  #returns a 3-tuple with the images corresponding to the index idx 
        """
        :param idx:
        :return:

        NEED TO MODIFY THE DIMENSION
        images          3D Tensor      (temporal * 3)   *   height(368)   *   weight(368)
        label_map       4D Tensor      temporal         *   label_size(45)   *   label_size(45) * joints
        center_map      3D Tensor      1                *   height(368)   *   weight(368)
        imgs            list of image directory
        """
        label_size = self.width // 8 - 1        # 45
        # counter = 1
        imgs = self.temporal_dir[idx]           # ['datadir/L0005.jpg', 'datadir/L0011.jpg', ... ]
        imgs.sort()
        # label_path = os.path.join(self.label_dir)
        if self.label_dir:
            labels = json.load(open(self.label_dir))    # has all the labels dict containing the 21-tuple of the joints for each key as the image name   
            # print(len(labels.keys()))
        # initialize
        images = np.zeros((self.width, self.height, self.temporal * 3))
        if self.label_dir:
            label_maps = np.zeros((self.temporal, label_size, label_size, self.joints))

        for i in range(self.temporal):          # get temporal images
            img = imgs[i]                       # 'datadir/L0005.jpg'

            # get image
            im = Image.open(img)              # read image of openCV 
            w, h, _ = np.asarray(im).shape     # weight 256 * height 256 * _
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)    # 368 / 256 = 1.4375

            im = im.resize((self.width, self.height))       # unit8      widht 368 * height 368 * 3
            images[:, :, (i * 3):(i * 3 + 3)] = self.normalize(im)   # float type 3D Vector  normalized height 368 * weight 368 * 3

            # get label map
            img_num = img.split('/')[-1][1:5]        # '0005'
            # print(img_num)

            if self.label_dir:
                # print('yes_before')
                # print(img_num)
                if img_num in labels: # for images without label, set label to zero
                    # print('yes')
                    # print(counter)
                    # counter+=1
                    label = labels[img_num]         #list       21 * 2
                    label_maps[i, :, :, :] = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)


        # generate the Gaussian heat map
        center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21, size_w=self.width, size_h=self.height)
        center_map = np.expand_dims(center_map,2)    #numpy of size 368*368*1

        if self.label_dir:
            return images.astype(np.float32), label_maps.astype(np.float32), center_map.astype(np.float32)
        else:
            return images.astype(np.float32), center_map.astype(np.float32)

        

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    #checked
    def genLabelMap(self, label, label_size, joints, ratio_x, ratio_y):
        """
        generate label heat map
        :param label:               list            21 * 2
        :param label_size:          int             45
        :param joints:              int             21
        :param ratio_x:             float           1.4375
        :param ratio_y:             float           1.4375
        :return:  heatmap           numpy           joints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros(( label_size, label_size, joints))
        background = np.zeros((label_size, label_size))

        # each joint
        for i in range(len(label)):             # length of the label = 21
            lbl = label[i]                      # [x_i, y_i]
            x = lbl[0] * ratio_x / 8.0          # scale the label to 46*46
            y = lbl[1] * ratio_y / 8.0
            heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy 46*46
            background += heatmap               # numpy
            label_maps[ :, :, i] = np.transpose(heatmap)

        return label_maps  # numpy           label_size * label_size * joints


# test case
'''

if __name__ == '__main__':
    temporal = 5
    data_dir = './001L0/'
    label_dir = './labels/001L0.json'
    outclass = 21    # must be the same as during trainin
    save_dir = './imput_check/'
    dataset = Feeder(data_dir=data_dir,label_dir=label_dir, temporal=temporal,train=True)

    a = dataset.temporal_dir
    # images, label_maps,center_map =  dataset[2]
    data = dataset[2]
    print(len(data))
    labels = data[1]
    print(labels.shape)

    lab = np.zeros((45,45))
    label = np.zeros((50,temporal*50))
    for j in range(temporal):
        lab = np.zeros((45,45))
        for k in range(outclass):
            lab += labels[j][:, :, k]
        label[0:45,45*j:45*j+45] = lab
    
    scipy.misc.imsave(save_dir + '/' + 'label_mod.jpg',label)

    # print(images.shape)  # (5*3) * 368 * 368)
    # print(label_maps.shape)  # 5 21 45 45
    # print(a[0].shape)
    '''
