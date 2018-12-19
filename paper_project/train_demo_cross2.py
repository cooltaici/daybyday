from FCN4_inception_ssd import *
from image_generator_ssd_pro import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import *
from keras_ssd_loss import *
from ssd_box_encode_decode_utils import *
import matplotlib.pyplot as plt
from FCN8 import *

BATCH = 23
aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
						[0.5, 1.0, 2.0]]
# aspect_ratios_per_layer=[[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
# 						[1.0/3.0, 0.5, 0.7, 1.0, 1.0/0.7, 2.0, 3.0],
# 						[1.0/3.0, 0.5, 1.0, 2.0, 3.0]]
scales=[ 0.15, 0.27, 0.39, 0.55]     #以224为基准
#scales=[ 0.14, 0.23, 0.34, 0.47]     #以224为基准
#scales=[ 0.35, 0.26, 0.17, 0.1]

normalize_coords = True
ImageSize = (224,224)
limit_boxes = True
two_boxes_for_ar1 = True
coords = 'centroids'
variances=[0.1, 0.1, 0.2, 0.2]
classes = ["background","tumor"]
nCross = 10
#加载网络
for crossID in range(nCross-1):
	SAVEPATH = r"E:\360brow\location2\brest\weight_save\crop\fcn8_100epoch_1024_Cross_{}_ssd.h5".format(crossID)
	PRE_WEIGHT = r"E:\360brow\location2\brest\weight_save\crop\fcn8_100epoch_1024_Cross_{}.h5".format(crossID)
	train_txt_path = r"E:\360brow\location2\brest\weight_save\crop\Cross_train_{}.txt".format(crossID)
	test_txt_path = r"E:\360brow\location2\brest\weight_save\crop\Cross_test_{}.txt".format(crossID)
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
	#model.load_weights(SAVEPATH)

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
	train_filenames,train_labels = generator0.parse_xml(images_paths=[r"E:\360brow\location2\brest\main_2000_jpg"],
														  annotations_paths=[r"E:\360brow\location2\brest\xml_annotation"],
														  image_set_paths=[train_txt_path],
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
	test_filenames,test_labels = generator1.parse_xml(images_paths=[r"E:\360brow\location2\brest\main_2000_jpg"],
													 annotations_paths=[r"E:\360brow\location2\brest\xml_annotation"],
													 image_set_paths=[test_txt_path],
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
							 min_delta=0.0001,
							 patience=5)
	def lr_schedule(epoch):
		if epoch <= 30: return 0.001
		else: return 0.0001
	#SSD
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0)
	#ours
	# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
	# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=10)
	model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

	result = model.fit_generator(generator = train_generator,
								  steps_per_epoch = round(len(train_filenames)/BATCH),
								  epochs = 100,
								  callbacks = [
											   LearningRateScheduler(lr_schedule)],
								  validation_data = test_generator,
								  validation_steps = round(len(test_filenames)))

	val_loss = result.history['val_loss']  # list类型
	loss = result.history['loss']

	open('sdd_loss.txt', 'a+').write(SAVEPATH + '\n')
	for kk in range(len(val_loss)):
		if kk % 2 == 0:
			if kk > 0:  #
				open('fcn8_loss.txt', 'a+').write(str(kk) + ':' + str(loss[kk]) + '   ;')
				open('fcn8_loss.txt', 'a+').write(str(kk) + ':' + str(val_loss[kk]) + '\n')

	print("training is over!")
	#测试一下结果，不用存储
	model.save_weights(SAVEPATH)
