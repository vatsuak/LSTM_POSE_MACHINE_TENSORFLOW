import os
from util import *
import tensorflow as tf
import pandas as pd
import model
import numpy as np
import os
from DataLoader import *


# hyper parameter
T = 3           # must be the same as during training
outclass = 21    # must be the same as during trainin
epoch = 99       # the epoch number of the model to load
save_dir = './predictions/'
data_dir = './001L0/'
model_dir = './ckpt/'
model_dir = os.path.join(model_dir,'lstm_pm_epoch{}.ckpt'.format(epoch))
batch_size = 1
label_dir = './labels/001L0.json'

# load data
#dataset = Feeder(data_dir=data_dir,label_dir=label_dir, train=False, temporal=T, joints=outclass)
dataset = Feeder(data_dir=data_dir,label_dir=None, train=False, temporal=T, joints=outclass,sigma = 1)
dl = DataLoader(dataset,batch_size,shuffle=False)
print('Dataset Loaded')


# hyper parameter


# load data
 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# **************************************** test all images ****************************************


# print('********* test data *********')


#placeholder for the image
image = tf.placeholder(tf.float32,shape=[None,368,368,T*3],name='temporal_info')

# placeholder for the gaussian
cmap = tf.placeholder(tf.float32,shape=[None,368,368,1],name='gaussian_peak')

# placeholder for the dropout probability
# dropprob = tf.placeholder(tf.float32,name='dropout')

#load the model
net = model.Net(outclass=outclass,T=T)

# create the graph for the feed forwatd network
predict_heatmaps = net.forward(image,cmap)


                             #****************BUILDING THE GRAPH*********************

saver = tf.train.Saver()

with tf.Session() as sess:
        
    #restore the model

    print('.............................................Restoring model.....................................')

    saver.restore(sess,model_dir)

    print('.............................................Saving Predictions.....................................')

    for step in range(len(dl)//batch_size):

    # get the inputs for the placeholders
        images, center = dl()
        # label = cv2.blur(label,(1,1,5,5,1))
    # images = np.full((1,368,368,T*3), 1.0)
    # center = np.full((1,368,368,1), 1.0)
    # label = np.full((1,T,46,46,outclass),1.0)

        # get the prediction from the saved model
        prediction = sess.run(predict_heatmaps,feed_dict={image:images,cmap:center})

    #no gradient calculation so no need to run trainer
    
    

        #ignoring the initial heatmap(used as a prior)
        prediction =  prediction[1:]
        
        #pred_images2(prediction, label,step, temporal=T, save_dir=save_dir)
        pred_images(prediction, step,temporal=T, save_dir=save_dir)



