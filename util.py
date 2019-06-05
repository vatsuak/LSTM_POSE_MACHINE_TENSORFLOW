import numpy as np
import os
import scipy.misc
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
import cv2

# the predict heatmaps come in as a list of tensors the remaining are normal tensors, so converting the predictions also to be tensors completely 
def calc_loss(predict_heatmaps, label_map, criterion, temporal):
    '''
    :param prediction(predict_heatmap(list of size (temporal+1)*[batch_size,46,46,21])), 
    :param label_map the groundtruth labels
    return:
    total_loss
    '''
    # print(predict_heatmaps.get_shape())

    predict = predict_heatmaps[0]  # the initial prediction 

    # print(label_map.get_shape())

    target = label_map[:, 0, :, :, :]
    initial_loss = criterion(target,predict)  # loss initial
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t+1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(target,predict)  # loss in each stage
        total_loss += tmp_loss
    return total_loss



#change to tf operations
def lstm_pm_evaluation(label_map, predict_heatmaps, sigma, temporal):
    pck_eval = []
    empty = np.zeros((45, 45, 21))                                      # 3D numpy  46 * 46 * 21
    for b in range(label_map.shape[0]):        # for each batch (person)
        for t in range(temporal):           # for each temporal
            target = label_map[b, t, :, :, :]     # 3D numpy  46 * 46 * 21
            predict = predict_heatmaps[t][b, :, :, :]  # 3D numpy  46 * 46 *21
            if not np.equal(empty, target).all():
                pck_eval.append(PCK(predict, target,label_size =45, sigma=sigma))

    return sum(pck_eval) / float(len(pck_eval))  #


#change to tf operations
def PCK(predict, target, label_size, sigma):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       46 * 46 * 21
    :param target:          3D numpy       46 * 46 * 21
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(predict.shape[2]):
        pre_x, pre_y = np.where(predict[:, :, i] == np.max(predict[:, :, i]))
        tar_x, tar_y = np.where(target[:, :, i] == np.max(target[:, :, i]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1
    return pck / float(predict.shape[2])




#given a series heatmaps, we display the most possible position of the joints
#modification of the save image function in the utils


def pred_images(heatmaps,step,temporal, save_dir):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal *  46 * 46 *21
    :param step: which batch number it is
    :param temporal:
    :return:
    """
    b=0
    output = np.zeros((50 , 50 * temporal))           # cd .. temporal save a single image
    for t in range(temporal):                           # for each temporal
        pre = np.zeros((45, 45))  #
        for i in range(21):                             # for each joint
            pre += heatmaps[t][b, :, :, i]  # 2D
        #_,pre = cv2.threshold(pre,0.5,1,cv2.THRESH_BINARY)
        output[0:45,  50 * t: 50 * t + 45] = pre

        if not os.path.exists(save_dir ):
            os.mkdir(save_dir )
    #output = scaler_model.fit_transform(output)
    scipy.misc.imsave(save_dir + '/' + str(step) + '.jpg', output[0:44])

def pred_images2(heatmaps,label_map,step,temporal, save_dir):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal *  46 * 46 *21
    :param step: which batch number it is
    :param temporal:
    :return:
    """
    b=0
    # thresh = .5
    output = np.ones((100 , 50 * temporal))           # cd .. temporal save a single image
    for t in range(temporal):                           # for each temporal
        pre = np.zeros((45, 45))  #
        gth = np.zeros((45, 45))
        for i in range(21):
            # print(t,b,i)
            # print(heatmaps[t][b, :, :, i])                             # for each joint
            pre += heatmaps[t][b, :, :, i]  # 2D
            gth += label_map[b][t, :, :, i]
        # _,pre = cv2.threshold(pre,thresh,1,cv2.THRESH_BINARY)
        # _,gth = cv2.threshold(pre,.1,1,cv2.THRESH_BINARY)        
        # output[0:45,  50 * t: 50 * t + 45] = cv2.blur(pre,(5,5))
        output[0:45,  50 * t: 50 * t + 45] = scaler_model.fit_transform(pre)

        output[50:95, 50 * t: 50 * t + 45] = gth

        if not os.path.exists(save_dir ):
            os.mkdir(save_dir )
    # output = scaler_model.fit_transform(output)
    # output = cv2.blur(output,(8,8))
    # scipy.misc.imsave(save_dir + '/' + str(step) + '.jpg', cv2.blur(output,((int(thresh*10),int(thresh*10)))))
    scipy.misc.imsave(save_dir + '/' + str(step) + '.jpg', output)
