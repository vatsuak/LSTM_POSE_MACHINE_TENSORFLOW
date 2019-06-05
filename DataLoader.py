from data_feeder import Feeder
import random
import numpy as np

class DataLoader():

    def __init__(self, feedob, batch_size, shuffle ):

        self.feedob = feedob
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.length = len(self.feedob.temporal_dir)
        self.i = 0
        if shuffle:
            random.shuffle(self.feedob.temporal_dir)

    def __len__(self):
        return self.length

    def __call__(self):
        img_batch = np.zeros((self.batch_size, 368, 368, self.feedob.temporal*3))
        if self.feedob.label_dir:
            label_batch = np.zeros((self.batch_size, self.feedob.temporal, 45, 45, self.feedob.joints))
        cmap_batch = np.zeros((self.batch_size, 368,368,1))

        for bval,ival in enumerate(range(self.i,self.i+self.batch_size)):
            img_batch[bval]= self.feedob[ival% self.length][0]
            if self.feedob.label_dir:
                label_batch[bval]= self.feedob[ival% self.length][1]
                cmap_batch[bval]= self.feedob[ival% self.length][2]
            else:
                cmap_batch[bval]= self.feedob[ival% self.length][1]


            # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        self.i = (self.i + self.batch_size) % self.length
        if self.feedob.label_dir:
            return img_batch, label_batch, cmap_batch
        else:
            return img_batch, cmap_batch

# if __name__ == '__main__':
#     temporal = 3
#     data_dir = './data/'
#     label_dir = './labels/001L0.json'

#     dataset = Feeder(data_dir=data_dir, label_dir=None, temporal=temporal,train=True)
#     dl = DataLoader(dataset,5)

#     # print(dl()[2][0])
#     for i in range(len(dl)//5+1):
#         print(dl.i)
#         j=dl()