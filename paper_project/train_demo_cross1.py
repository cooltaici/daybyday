from deep_learning_models import *
from FCN8 import *
from RefineNet.RefineNet import *
from DeepLabv3plus.DeepLabplus import *
from PSPNet.PSPNet import *
from image_generator import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow as tf
from ssd_box_encode_decode_utils import *
from keras.callbacks import Callback

BATCH = 12
nCross = 10

for crossID in range(0,nCross):
    epochs = 150
    # if crossID==0:
    #     epoch = 70
    #model = FCN8(nClasses=128)
    model = build_refinenet((224, 224, 3), 1, None, "conv")
    #model = build_pspnet(1, 101, (224, 224))
    # model = Deeplabv3(weights=None, input_tensor=None, input_shape=(224, 224, 3), classes=1, backbone='xception',
    #           OS=8, alpha=1.)
    model.summary()

    train_txt_path = r"E:\360brow\location2\brest\weight_save\crop\Cross_train_{}.txt".format(crossID)
    test_txt_path = r"E:\360brow\location2\brest\weight_save\crop\Cross_test_{}.txt".format(crossID)
    #model_save_path = r"E:\360brow\location2\brest\weight_save\crop\fcn8_1024_Cross_{}_maxvalac.h5".format(crossID)
    model_save_path = r"E:\360brow\location2\brest\weight_save\crop\RefineNet_224_batch12_Cross_{}.h5".format(crossID)
    if os.path.exists(model_save_path):
        model.load_weights(model_save_path)
    optimizers3 = optimizers.SGD(lr=0.008, momentum=0.92, nesterov=True)

    # def lr_schedule(epoch):
    #     if epoch >60:
    #         return 0.008
    #     if epoch >30:
    #         return 0.01
    #     return 0.03
    def lr_schedule(epoch):
        power = 0.9
        maxepoch = epochs
        return 0.015*pow(1 - epoch / maxepoch, power)

    model.compile(loss=loss2_4,  # loss2_4,focal_loss（有问题）
                  optimizer=optimizers3,
                  metrics=['binary_accuracy'])  # pro最好作为定位使用,acc_image,binary_accuracy

    # model.load_weights(r"E:\360brow\location2\brest\weight_save\test\fcn8_50epoch_1024_9822.h5")

    generator0 = BatchGenerator()
    train_filenames,train_Mask_filenames = generator0.parse_xml(images_paths=[r"E:\360brow\location2\brest\main_2000_jpg"],
                                                                mask_paths=[r"E:\360brow\location2\brest\main_2000_mask_bmp"],
                                                                image_set_paths=[train_txt_path],ret = True)
    train_generator = generator0.generate(
                                         batch_size=BATCH,
                                         gama=True,
                                         resize = [224, 224],
                                         zoom_range=[0.9,1.20],
                                         rotation_range = 18,
                                         width_shift_range = 0.22,
                                         height_shift_range = 0.22,
                                         horizontal_flip = True,
                                         train = True)
    # 监控设置
    check_point = ModelCheckpoint(model_save_path,
                                  monitor='val_binary_accuracy',  #val_binary_accuracy
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='max',
                                  period=1)
    stop_set = EarlyStopping(monitor='val_binary_accuracy',
                             min_delta=0.0001,
                             patience=10)

    generator1 = BatchGenerator()
    test_filenames,test_Mask_filenames = generator1.parse_xml(images_paths=[r"E:\360brow\location2\brest\main_2000_jpg"],
                                                              mask_paths=[r"E:\360brow\location2\brest\main_2000_mask_bmp"],
                                                              image_set_paths=[test_txt_path],ret=True)
    test_generator = generator1.generate(batch_size=1,train = True,Transform=False)            #测试样本不作处理

    result = model.fit_generator(train_generator,
                                 steps_per_epoch = len(train_filenames)//BATCH,
                                 validation_data=test_generator,
                                 validation_steps=len(test_filenames),
                                 callbacks=[LearningRateScheduler(lr_schedule)], #savebest  [check_point,stop_set],check_point
                                 epochs = epochs,
                                 verbose = 1)
    model.save_weights(model_save_path)
    val_loss = result.history['val_loss']  # list类型
    loss = result.history['loss']
    val_acc = result.history['val_binary_accuracy']

    open('fcn8_loss.txt', 'a+').write(model_save_path + '\n')
    for kk in range(len(val_loss)):
        if kk % 2 == 0:
            if kk > 0:  #
                open('fcn8_loss.txt', 'a+').write(str(kk) + ':' + str(val_acc[kk]) + ';   ')
                open('fcn8_loss.txt', 'a+').write(str(kk) + ':' + str(loss[kk]) + ';   ')
                open('fcn8_loss.txt', 'a+').write(str(kk) + ':' + str(val_loss[kk]) + '\n')

    #model.save_weights(model_save_path)
    print("Cross {} is finished!".format(crossID))
    break
print("over")

