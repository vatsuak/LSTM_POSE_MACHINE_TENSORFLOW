# test
import os
from util import *
import tensorflow as tf
import pandas as pd
import model
import numpy as np
import os
from DataLoader import *


# hyper parameter
T = 5           # must be the same as during training
outclass = 21    # must be the same as during trainin
epoch = 0       # the epoch number of the model to load
save_dir = './validation_info/'
data_dir = './001L0/'
label_dir = './labels/001L0.json'
model_dir = './ckpt/2'
model_dir = os.path.join(model_dir,'lstm_pm_epoch{}.ckpt'.format(epoch))
batch_size = 5


# load data
dataset = Feeder(data_dir=data_dir, label_dir=label_dir, train=False, temporal=T, joints=outclass)
dl = DataLoader(dataset,batch_size,shuffle=False)
print('Dataset Loaded')
 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# **************************************** test all images ****************************************


# print('********* test data *********')


#placeholder for the image
image = tf.placeholder(tf.float32,shape=[None,368,368,T*3],name='temporal_info')
    
# the output prediction should come out as 46*46*21
# label_map = tf.placeholder(tf.float32,shape=[None,T,46,46,outclass])

# placeholder for the gaussian
cmap = tf.placeholder(tf.float32,shape=[None,368,368,1],name='gaussian_peak')

# placeholder for sigma
# sigma = tf.placeholder(tf.float32,name='sigma')

# placeholder for the dropout probability
dropprob = tf.placeholder(tf.float32,name='dropout')

# placeholder for the predicted heatmap
# pred = tf.placeholder(tf.float32,shape=[T,None,46,46,outclass])

#load the model
net = model.Net(outclass=outclass,T=T,prob=dropprob)

# create the graph for the feed forwatd network
predict_heatmaps = net.forward(image,cmap)





                             #****************BUILDING THE GRAPH*********************



saver = tf.train.Saver()

sigma = 0.01
results = []

with tf.Session() as sess:
        
    #restore the model

    print('.............................................Restoring model.....................................')

    saver.restore(sess,model_dir)

    for i in range(5): #going over the sigmas

    #modify into the sessions process
        result = []  # save sigma and pck
        result.append(sigma)
        pck_all = []
        for step in range(len(dl)//batch_size):

        # get the inputs for the placeholders
            images, label, center = dl()

        # images = np.full((1,368,368,T*3), 1.0)
        # center = np.full((1,368,368,1), 1.0)
        # label = np.full((1,T,46,46,outclass),1.0)

            # get the prediction from the saved model
            prediction = sess.run(predict_heatmaps,feed_dict={image:images,cmap:center,dropprob:1.0})

        #no gradient calculation so no need to run trainer
        
        

            #ignoring the initial heatmap(used as a prior)
            prediction =  prediction[1:]

            # calculate pck
            pck = lstm_pm_evaluation(label, prediction, sigma=sigma, temporal=T)

            pck_all.append(pck)


        print('sigma ==========> ' + str(sigma))
        print('===PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all)))
        result.append(str(sum(pck_all) / len(pck_all)))
        results.append(result)

        sigma += 0.01

    results = pd.DataFrame(results)
    results.to_csv('ckpt/' + 'test_pck.csv', header=['Sigma','Avg. Pck'], index=None,sep='\t')
