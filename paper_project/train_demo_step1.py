#from deep_learning_models import *
from fcn_inception import *
from FCN8 import *
from image_generator import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from fcn8_inception2 import *
import tensorflow as tf
import keras.backend as K
from keras import metrics
from ssd_box_encode_decode_utils import *
from keras.callbacks import Callback


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

# def IOU(y_true, y_pred):
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#     sa1 = (y_true>0.5)
#     sa2 = (y_pred>0.5)
#     insection = K.equal(sa1,sa2)
#     union = sa1+sa2
#     union = (union>0)
#     insection = K.cast(insection,np.float32)
#     union = K.cast(union,np.float32)
#     insection = K.sum(insection,axis=-1)
#     union = K.sum(union,axis=-1)
#     return (insection+1.0)/(union+1.0)

# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed
def focal_loss(y_true, y_pred):
    gamma=2.
    alpha=.25
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    #shape = K.eval(K.shape(y_true))
    shape = K.shape(y_true)
    y_true = K.reshape(y_true,(-1,shape[1]*shape[2]))
    y_pred = K.reshape(y_pred,(-1,shape[1]*shape[2]))
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1),axis=-1)-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0),axis=-1)
#二维对数损失
def loss2_4(y_true, y_pred):
    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)
    shape = K.shape(y_true)
    y_true = K.reshape(y_true,(-1,shape[1]*shape[2]))
    y_pred = K.reshape(y_pred,(-1,shape[1]*shape[2]))
    xent_loss = metrics.binary_crossentropy(y_true, y_pred)  #为什么使用K后端就不行呢？
    return xent_loss

BATCH = 5
SAVEPATH = r"E:\data_base\breast\weight_save\crop\fcn8_1024_Cross_1.h5"
generator0 = BatchGenerator()
train_filenames,train_Mask_filenames = generator0.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
                                                            mask_paths=[r"E:\data_base\breast\main_2000_mask_bmp"],
                                                            image_set_paths=[r"E:\data_base\breast\weight_save\crop\Cross_test_0.txt"],ret = True)
train_generator = generator0.generate(
                                     batch_size=BATCH,
                                     #brightness = [0.5,0.7,0.5],
                                     resize = [224, 224],
                                     zoom_range=[0.9,1.20],
                                     rotation_range = 18,
                                     width_shift_range = 0.22,
                                     height_shift_range = 0.22,
                                     horizontal_flip = True,
                                     train = True)

generator1 = BatchGenerator()
test_filenames,test_Mask_filenames = generator1.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
                                                          mask_paths=[r"E:\data_base\breast\main_2000_mask_bmp"],
                                                          image_set_paths=[r"E:\data_base\breast\weight_save\crop\Cross_test_0.txt"],ret=True)
test_generator = generator1.generate(batch_size=BATCH,train = True,Transform=False)            #测试样本不作处理

#model = fcn4_inception3()
#model = fcn8_inception2(isnorm=True)
model = FCN8(128)
model.load_weights(SAVEPATH)

model.summary()
optimizers3 = optimizers.SGD(lr=0.02, momentum=0.92, nesterov=True)

# model_save = r'model_save\Sigment_fcn4_inception_weights_strain'
# if os.path.exists(model_save):
#     model.load_weights(model_save)

model.compile(loss = focal_loss,  #loss2_4,focal_loss
              optimizer = optimizers3,
              metrics = ['binary_accuracy'])  #pro最好作为定位使用,acc_image,binary_accuracy

#监控设置
check_point =   ModelCheckpoint(SAVEPATH,
                               monitor='val_acc', #"val_loss"
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                               mode='max',
                               period=3)
stop_set = EarlyStopping(monitor='val_loss',  #"val_loss"
                         verbose=1,
                         mode="min",
                         min_delta=0.0001,
                         patience=18)  #18个epoch没改善就停止训练

# savebest = ModelCheckpoint_one(filepath=SAVEPATH,
#                                save_weights_only=True)

# result = model.fit_generator(train_generator,
#                              steps_per_epoch = len(train_filenames)//BATCH,
#                              validation_data=test_generator,
#                              validation_steps=len(test_filenames)//BATCH,
#                              #callbacks=[check_point,stop_set],   # [check_point,stop_set] savebest
#                              epochs = 120,
#                              verbose = 1)

#model.save_weights(r"")
model.load_weights(SAVEPATH)

#存储测试结果
generator1 = BatchGenerator()
test_filenames,test_Mask_filenames = generator1.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
                                                         mask_paths=[r"E:\data_base\breast\main_2000_mask_bmp"],
                                                         image_set_paths=[r"E:\data_base\breast\weight_save\crop\Cross_test_1.txt"],ret=True)
test_generator = generator1.generate(batch_size=1,train = False)            #测试样本不作处理

for i in range(len(test_filenames)):
    Img_Save_Path = r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\cross1\map"
    Xml_Save_Path = r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\cross1\temp"
    X, Y, original_images, original_labels,this_filenames = next(test_generator)

    index = 0
    XX,name = os.path.split(this_filenames[index])
    ImgID = name.split(".")[0]
    #得出输出
    Imgmap = model.predict(np.reshape(X[index],(-1,224,224,3)))
    pre = np.reshape(Imgmap[index],(224,224))*255.0
    #存储概率图
    img = Image.fromarray(pre.astype(np.uint8))
    img = img.resize((original_images[index].shape[1],original_images[index].shape[0]))
    img.save(os.path.join(Img_Save_Path,ImgID+".jpg"))
    #显示？
    #存储xml文件
    pred_box = clcbox_from_map_all(np.array(img),marjin=10)
    write_box_to_xml(pred_box,Xml_Save_Path,filename=ImgID+".xml",width=original_images[index].shape[1],height=original_images[index].shape[0],depth=3)
    print(ImgID+"have been saved")
print('over')

