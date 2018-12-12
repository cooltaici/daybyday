from keras.models import  load_model
import warnings
import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from skimage import exposure
from keras import metrics,losses,optimizers
import os,shutil
from PIL import Image
import matplotlib.pyplot as plt
#二级
def loss2_1(y_true, y_pred):  #数值非常大，为什么？
    y_true = K.flatten(y_true)                   #也可以使用K.reshape
    y_pred = K.flatten(y_pred)
    #y_pred = K.softmax(y_pred)
    xent_loss = metrics.categorical_crossentropy(y_true, y_pred)
    return xent_loss
    #return K.categorical_crossentropy(y_pred, y_true)

#二维重合率损失
def loss2_2(y_true, y_pred):
    mask_pred = K.cast(K.less(0.5, y_pred), K.floatx())
    tmp1 = K.sum(K.sum(mask_pred * y_true,axis=-1),axis=-1)
    tmp2 = K.sum(K.sum(mask_pred,axis=-1),axis=-1) + K.sum(K.sum(y_true,axis=-1),axis=-1) + K.epsilon()
    return 1.0 - 2 * tmp1 / tmp2

#二维均方差损失，速度较慢，不可取
def loss2_3(y_true, y_pred):  #没什么大问题
    return K.mean(K.mean(K.square(y_pred - y_true), axis = -1),axis = -1)

#二维对数损失
def loss2_4(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    #xent_loss = data_size[0] * data_size[1] * metrics.binary_crossentropy(y_true, y_pred)
    xent_loss = metrics.binary_crossentropy(y_true, y_pred)  #为什么使用K后端就不行呢？
    return xent_loss
#支持batch
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersec = y_true_f * y_pred_f
    intersection = K.sum(K.sum(intersec,axis=-1),axis=-1)
    return (2. * intersection + 1) / (K.sum(K.sum(y_true_f,axis=-1),axis=-1) + K.sum(K.sum(y_pred_f,axis=-1),axis=-1) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ((2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1))

#扩增预测函数
def preclass(pre):
    maxd = max(pre)
    classth = -1
    maxnum = 0
    for i in range(maxd+1):
        num = sum(pre==i)
        if num>maxnum:
            maxnum = num
            classth = i
    return classth


#预处理函数：能不能让标签的不发生改变呢
def add_gama(img):  #小于1的时候是变亮，大于1的时候是变暗
    choice = np.random.random()
    if choice>0.7:
       sed = np.random.random()*0.7
       return exposure.adjust_gamma(img, sed)
    if choice<=0.7 and choice>=0.2:
       sed = np.random.random()*0.8+0.7
       return exposure.adjust_gamma(img, sed)
    if choice<=0.2:
       sed = np.random.random()*0.8+1.5
       return exposure.adjust_gamma(img, sed)

def add_scale(img):        #对图像的拉伸变换,可以避免对标签的改变
    l1 = np.random.random()*0.2           #下边界
    l2 = np.random.random()*0.2          #上边界
    l2 = 1.0-l2
    img1 = exposure.rescale_intensity(img, out_range=(l1, l2))   #
    img1[img==0] = 0.0
    img1[img==1] = 1.0
    return img1

#直方图均衡化
def hist_transform(img):
    return exposure.equalize_hist(img)
#局部直方图均衡化
def adapthist_transform(img):
    x = exposure.equalize_adapthist(img, clip_limit=0.03)
    return x

#自定义准确率函数,测试成功，支持batch
def acc_image(y_true, y_pred):
    thre = 0.5
    sa1 = (y_true>0.5)
    sa2 = (y_pred>0.5)
    eq = K.equal(sa1,sa2)
    eq = K.cast(eq,np.float32)
    eqcount = K.sum(K.sum(eq,axis=-1),axis=-1)
    shape = K.shape(eq)
    count = shape[-1]*shape[-2]
    count = K.cast(count,np.float32)
    return eqcount/count

if __name__ == '__main__':
    #im = Image.open('E:\\图像数据库\\2017暑期课题\\正方形裁剪_126+\\分割测试\\原图总库\\711143_0.jpg')
    #im = Image.open('E:\\图像数据库\\2017暑期课题\\正方形裁剪_126+\\分割测试\\分割总库\\711143_0.jpg')
    # im = np.array(im)/255.0
    # im = (im>=0.5)
    # im = im.astype(np.float32)
    sa1 = np.random.random((3,6,6))
    sa2 = np.random.random((3,6,6))
    tensor1 = K.variable(sa1)
    tensor2 = K.variable(sa2)
    tensor3 = acc_image(tensor1, tensor2)
    print('load successfully')

#集成callback类,只存储最优解:不用频繁存储，速度会快一些
class ModelCheckpoint_one(Callback):
    def __init__(self, filepath = None, verbose=1,save_weights_only = True,nepoches = 3):
        super(ModelCheckpoint_one, self).__init__()
        self.monitor = 'val_categorical_accuracy'
        self.verbose = verbose
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = 0
        self.monitor_op = np.greater
        self.best = -np.Inf

    def on_epoch_end(self,epoch,logs=None):
        logs = logs or {}
        filepath = None
        if self.filepath is not None:
            filepath = self.filepath.format(epoch=epoch, **logs)
        current = logs.get(self.monitor)
        if self.period == 0:
           self.best = current
           self.period = self.period + 1
           return
        if current is None:
            warnings.warn('Can save best model only with categorical_accuracy available!', RuntimeWarning)
        else:
            self.period = self.period + 1
            if current > self.best:
                if self.verbose>0:
                    print('------------------->Epoch %d model are best!\n'%epoch)
                    print('当前最佳：',current)
                self.best = current
                #
                if self.filepath != None:
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                        #self.modellist[index].save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
            # else:
            #     if self.verbose > 0:
            #         print('Epoch %d: acc did not improve' %epoch)