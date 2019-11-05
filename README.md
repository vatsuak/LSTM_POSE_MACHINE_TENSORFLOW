# LSTM_POSE_MACHINE_TENSORFLOW
*Tensorflow implementation of [LSTM Pose Machines](https://arxiv.org/pdf/1712.06316.pdf)*

## Prerequisites
* Python 3.5
* tensorflow-gpu 1.12
* scipy
* sklearn
* pillow
* pandas
* numpy

All the requiremenst can be installed by running:

```
pip install -r requirements.txt
```

Code based on *tensorflow-gpu 1.12*

## set hyper-parameters in 
> train.py

```
T = 5                            # how many timestamps to look back into 
outclass = 21                    # number of joints to be tracked
learning_rate = 8e-6
batch_size = 4                   # batch size* temporal must be atleast total number of images in the dataset otherwise the batches will be reported as the same images in a cyclic manner
epochs = 101
begin_epoch = 0
save_dir = './ckpt/'                        # to save model
data_dir = './001L0/'                        # the train data dir
label_dir = './labels/001L0.json'           # the label dir

save_dir_val= './validation_info/'          # dir to save the validation info i.e.the csv
data_dir_val = './001L00/'                # dir to find the validation dataset
label_dir_val = './labels/001L0.json'       # dir to find the validation labels should point to a json file
```

### To train and validate run:
```
python lstm_pm_train.py
```

### For prediction run:
```
python lstm_pm_prediction.py
```

### Dataset Credits
https://github.com/HowieMa/lstm_pm_pytorch

## References
[LSTM Pose Machines](https://arxiv.org/pdf/1712.06316.pdf)

[lawy623/LSTM_Pose_Machines](https://github.com/lawy623/LSTM_Pose_Machines)
