#model block
from layers import *
import tensorflow as tf

class Net():

    def __init__(self, outclass=21, T=7):
        self.outclass = outclass
        self.T = T
        # self.prob = prob

    def convnet1(self,img):
        '''
        denoted by F0(.) in the paper 
        :param img:  368 * 368 * 21
        :return: initial_heatmap  45 * 45 * out_class(21)
        '''
        with tf.name_scope('conv1'):
            x = pool2d(tf.nn.relu(conv(img,3,128,9)))
            x = pool2d(tf.nn.relu(conv(x,128,128,9)))
            x = pool2d(tf.nn.relu(conv(x,128,128,9)))
            x = tf.nn.relu(conv(x,128,32,5))
            x = tf.nn.relu(conv(x,32,512,9))
            x = tf.nn.relu(conv(x,512,512,1))
            # drop_x = tf.nn.dropout(x,keep_prob=self.prob)
            # initial_heatmap = conv(drop_x,512,self.outclass,1)
            initial_heatmap = conv(x,512,self.outclass,1)
            return initial_heatmap

    def convnet2(self,img):
        ''' denoted by F(.) in the paper
            :param img: 368 * 368 *3
            :return: Fs(.)  45 * 45 * features(32)
        '''
        with tf.name_scope('conv2'):
            x = pool2d(tf.nn.relu(conv(img,3,128,9)))
            x = pool2d(tf.nn.relu(conv(x,128,128,9)))
            x = pool2d(tf.nn.relu(conv(x,128,128,9)))
            x = tf.nn.relu(conv(x,128,32,5))
            return x

    def convnet3(self,hide_t):
        """  generator denoted by g(.) in the paper
            :param h_t: 45 * 45 * 48
            :return: heatmap   45 * 45 * outclass
        """
        with tf.name_scope('conv3'):
            x = tf.nn.relu(conv(hide_t,48,128,11))
            x = tf.nn.relu(conv(x,128,128,11))
            x = tf.nn.relu(conv(x,128,128,11))
            x = tf.nn.relu(conv(x,128,128,1))
            x = conv(x,128,self.outclass,1)
            return x
        
    def lstm0(self,x):
        ''' denoted by L(.) at t=1 in the paper
            :param x:  45 * 45 * (cat of initial heatmap(21), features(32) and centremap(1))
            :return:
            hide_t:    45 * 45 * 48
            cell_t:    45 * 45 * 48
        '''
        with tf.name_scope('LSTM_t1'):

            gx = conv(x,32+1+self.outclass,48,3)
            ix = conv(x,32+1+self.outclass,48,3)
            ox = conv(x,32+1+self.outclass,48,3)
            # there is nothing to forget so omit the forget gate

            gx = tf.nn.tanh(gx)
            ix = tf.nn.sigmoid(ix)
            ox = tf.nn.sigmoid(ox)

            cell1 = tf.nn.tanh(tf.multiply(gx,ix))
            hide_1 = tf.multiply(ox,cell1)
            return cell1, hide_1

    def lstm(self, heatmap, features, centermap, hide_t_1, cell_t_1):
        ''' denoted by L(.) at t>1 in the paper
            :param heatmap:     45 * 45 * output(21)
            :param features:    45 * 45 * 32
            :param centermap:   45 * 45 * 1
            :param hide_t_1     45 * 45 * 48
            :param cell_t_1:    45 * 45 * 48
            :return:
            hide_t:    45 * 45 * 48
            cell_t:    45 * 45 * 48
        '''
        with tf.name_scope('LSTM'):
            
            xt = tf.concat([heatmap, features, centermap], axis=3)  #  45 * 45 * (32+ class+1 +1 ) 

            gx = conv(xt,32+1+self.outclass,48,3,bias=True)  # output: 45 * 45 * 48
            gh = conv(hide_t_1,48,48,3,bias=False)  # output: 45 * 45 * 48
            g_sum = tf.add(gx, gh)
            gt = tf.nn.tanh(g_sum)

            ox = conv(xt,32+1+self.outclass,48,3,bias=True)  # output: 45 * 45 * 48
            oh = conv(hide_t_1,48,48,3,bias=False)  # output: 48 * 45 * 45
            o_sum = tf.add(ox, oh)
            ot = tf.nn.sigmoid(o_sum)

            ix = conv(xt,32+1+self.outclass,48,3,bias=True)  # output: 45 * 45 * 48
            ih = conv(hide_t_1,48,48,3,bias=False)  # output: 48 * 45 * 45
            i_sum = tf.add(ix, ih)
            it = tf.nn.sigmoid(i_sum)

            fx = conv(xt,32+1+self.outclass,48,3,bias=True)  # output: 48 * 45 * 45
            fh =conv(hide_t_1,48,48,3,bias=False)  # output: 48 * 45 * 45
            f_sum = tf.add(fx,fh)
            ft = tf.nn.sigmoid(f_sum)

            cell_t = tf.add(tf.multiply(ft,cell_t_1),tf.multiply(it,gt))
            hide_t = tf.multiply(ot,tf.nn.tanh(cell_t))

            return cell_t, hide_t

    def stage1(self, img, c_map):
        '''
            :param img:                368 * 368 * 3
            :param c_map:                 368 * 368 * 1
            :return:
            heatmap:                     45 * 45 * out_class
            cell_t:                      45 * 45 * 48
            hide_t:                      45 * 45 * 48
        '''
        with tf.name_scope('Stage1'):

            initial_heatmap = self.convnet1(img)
            features = self.convnet2(img)
            centermap = pool_center_lower(c_map)


            x = tf.concat([initial_heatmap, features, centermap], axis=3)
            cell1, hide1 = self.lstm0(x)
            heatmap = self.convnet3(hide1)
            return initial_heatmap, heatmap, cell1, hide1

    def stage2(self, img, c_map, heatmap, cell_t_1, hide_t_1):
        '''
            :param img:                368 * 368 * 3
            :param c_map: gaussian     368 * 368 * 1
            :param heatmap:            45 * 45 * outclass
            :param cell_t_1:           45 * 45 * 48
            :param hide_t_1:           45 * 45 * 48
            :return:
            new_heatmap:                45 * 45 *outclass
            cell_t:                     45 * 45 * 48
            hide_t:                     45 * 45 * 48
        '''
        
        with tf.name_scope('Stage2'):
            
            features = self.convnet2(img)
            centermap = pool_center_lower(c_map)
            cell_t, hide_t = self.lstm(heatmap, features, centermap, hide_t_1, cell_t_1)
            new_heat_map = self.convnet3(hide_t)
            return new_heat_map, cell_t, hide_t

    def forward(self, images,cmap):
        '''   

            :param images:      Tensor      w(368) * h(368) * (T * 3)
            :param center_map:  Tensor      368 * 368 * 1 
            :return:
            heatmaps            list        (T + 1)* 45 * 45 * out_class   includes the initial heatmap(initial + T other steps)
        '''

        with tf.name_scope('forward'):
            image = images[:, :, :, 0:3]
                
            heat_maps = []  # change to an empty tensor and add the value to that insted of a list change the loss calculation accordingly
            initial_heatmap, heatmap, cell, hide = self.stage1(image, cmap)  # initial heat map

            heat_maps.append(initial_heatmap)  # for initial loss
            heat_maps.append(heatmap)

            for i in range(1, self.T):
                image = images[:, :, :, (3 * i):(3 * i + 3)]
                heatmap, cell, hide = self.stage2(image, cmap, heatmap, cell, hide)
                heat_maps.append(heatmap)
            heat_maps = tf.stack(heat_maps)

            return heat_maps


if __name__ == '__main__':
    import numpy as np
    image = (np.random.rand(1,368,368,15)).astype(np.float32)
    cmap = (np.random.rand(1,368,368,1)).astype(np.float32)
    net = Net(21,5) 
    maps = net.forward(image,cmap)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        maps = sess.run(maps)
    for i in range(len(maps)):
        print(maps[i].shape)




