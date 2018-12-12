#from FCN4_inception_ssd import *
from image_generator_ssd_pro import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import *
from keras_ssd_loss import *
from ssd_box_encode_decode_utils import *
import matplotlib.pyplot as plt
from FCN8 import *

BATCH = 4
SAVEPATH = r"E:\数字图像处理\当前项目\paper_project\weight_save\fcn8_50epoch_1024_9822_a130_ssd.h5"

aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
						[0.5, 1.0, 2.0]]
# aspect_ratios_per_layer=[[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
# 						[1.0/3.0, 0.5, 0.7, 1.0, 1.0/0.7, 2.0, 3.0],
# 						[1.0/3.0, 0.5, 1.0, 2.0, 3.0]]
scales=[ 0.15, 0.27, 0.39, 0.55]     #以224为基准
#scales=[ 0.35, 0.26, 0.17, 0.1]

normalize_coords = True
ImageSize = (224,224)
limit_boxes = True
two_boxes_for_ar1 = True
coords = 'centroids'
variances=[0.1, 0.1, 0.2, 0.2]
classes = ["background","tumor"]
#加载网络
PRE_WEIGHT = r"E:\数字图像处理\当前项目\paper_project\weight_save\fcn8_50epoch_1024_9822_a130.h5"
model,predict_size = FCN8_ssd(weightpath=PRE_WEIGHT,
										isnorm = True,
										image_size = ImageSize,
										n_classes = 2,
										min_scale=None,
										max_scale=None,
										scale = scales,
										aspect_ratios_global=None,
										aspect_ratios_per_layer=aspect_ratios_per_layer,
										two_boxes_for_ar1=two_boxes_for_ar1,
										limit_boxes=limit_boxes,
										variances=variances,
										coords=coords,
										normalize_coords=normalize_coords)
model.summary()

ssd_box_encoder = SSDBoxEncoder(img_height=ImageSize[0],
                                img_width=ImageSize[0],
                                n_classes=2,
                                predictor_sizes = predict_size,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,              #和网络中设置一致
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios_per_layer,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords='centroids',
                                normalize_coords=normalize_coords)

generator0 = BatchGenerator()
train_filenames,train_labels = generator0.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
                                                      annotations_paths=[r"E:\data_base\breast\xml_annotation"],
                                                      image_set_paths=[r"E:\data_base\breast\train.txt"],
													  classes = classes,
													  ret = True)
train_generator = generator0.generate(
										batch_size=BATCH,
										train=True,
										Transform=True,
										ssd_box_encoder=ssd_box_encoder,
										equalize=True,
										gama=True,
										translate=((1, 20), (1, 20), 0.9),
										resize=(224, 224),
										gray=False,
										limit_boxes=True,
										include_thresh=0.4)


generator1 = BatchGenerator()
test_filenames,test_labels = generator1.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
												 annotations_paths=[r"E:\data_base\breast\xml_annotation"],
												 image_set_paths=[r"E:\data_base\breast\test.txt"],
												 classes = classes,
												 ret=True)
test_generator = generator1.generate(
									 batch_size=1,
									 train=True,
									 ssd_box_encoder=ssd_box_encoder,
									 Transform=False,
									 resize=(224,224))
#监控设置
check_point = ModelCheckpoint(SAVEPATH,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                               mode='auto',
                               period=1)
stop_set = EarlyStopping(monitor='val_loss',
                         min_delta=0.001,
                         patience=5)

model.summary()
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=10)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# history = model.fit_generator(generator = train_generator,
#                               steps_per_epoch = round(len(train_filenames)/BATCH),
#                               epochs = 200,
#                               callbacks = [check_point,
#                                            #LearningRateScheduler(lr_schedule),
#                                            stop_set],
#                               validation_data = test_generator,
#                               validation_steps = round(len(test_filenames)))
model.load_weights(r"E:\数字图像处理\当前项目\paper_project\weight_save\fcn8_50epoch_1024_9822_a130_ssd_100epo.h5")
print("training is over!")
#测试一下结果，不用存储
generator1 = BatchGenerator()
test_filenames,test_labels = generator1.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
												 annotations_paths=[r"E:\data_base\breast\xml_annotation"],
												 image_set_paths=[r"E:\data_base\breast\test.txt"],
												 classes = classes,
												 ret=True)
test_generator = generator1.generate(
									 batch_size=1,
									 train=False,
									 ssd_box_encoder=ssd_box_encoder,
									 resize=(224,224))

#有个任务，需要调节SSD预选框筛选的参数，使得最后留下来的检测框尽量少
for k in range(len(test_filenames)):
	Xml_Save_Path = r"E:\data_base\breast\map_save\fcn8_ssd\xml"
	X, y_true, Row_X, Row_y, this_filenames = next(test_generator)  #X是resize的图片，y_true是预选框
	index = 0
	XX,name = os.path.split(this_filenames[index])
	ImgID = name.split(".")[0]

	y_pred  = model.predict(X)
	y_temp = np.copy(y_true)
	y_temp[:,:,0:-12]=y_pred       #用预测值替换实际值
	#y_pred_decoded是list类型，
	y_pred_decoded = decode_y(y_temp,
							  confidence_thresh=0.5,		#用于预选框筛选
							  iou_threshold=0.80,          #用于非极大值抑制
							  top_k=1000,                     #我们的样本最多只有3个目标
							  input_coords='centroids',
							  normalize_coords=normalize_coords, #从归一化坐标恢复
							  img_height=ImageSize[0],
							  img_width=ImageSize[1])
	y_temp = y_pred_decoded[index]       #nbox*(2+4),            (background,tumor,coords)
	sort_index = np.argsort(y_temp[:,1])
	y_temp = y_temp[sort_index[-1::-1],:]      #按照肿瘤的置信度排序
	indexmax  = 0
	#ymax = y_temp[indexmax[1],:]

	#显示置信度最大的一个框
	plt.figure(1)
	plt.imshow(X[0])
	current_axis = plt.gca()
	for kk in range(y_temp.shape[0]):
		y_tempkk = y_temp[kk]
		label = '{}: {:.2f}'.format(classes[int(y_tempkk[0])], y_tempkk[1])
		current_axis.add_patch(plt.Rectangle((y_tempkk[2], y_tempkk[4]), y_tempkk[3]-y_tempkk[2], y_tempkk[5]-y_tempkk[4], color='blue', fill=False, linewidth=2))
		current_axis.text(y_tempkk[2], y_tempkk[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

	#存储xml文件，包含所有box,不需要存储
	# pred_box = y_temp[:,2:]
	# write_box_to_xml(pred_box,Xml_Save_Path,filename=ImgID+".xml",width=Row_X[index].shape[1],height=Row_X[index].shape[0],depth=3)
print('训练结束')

