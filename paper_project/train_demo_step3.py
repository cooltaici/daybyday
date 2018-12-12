
from FCN_advance2 import *
from image_generator_ssd import *

from ssd_box_encode_decode_utils import *
import matplotlib.pyplot as plt
BATCH = 30
#原来的不合适
# aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
# 						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
# 						[0.5, 1.0, 2.0]]
# scales=[ 0.15, 0.27, 0.39, 0.55]     #以224为基准
#advance0
# aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
# 						 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
# 						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
# 						[0.5, 1.0, 2.0]]
# scales=[ 0.07, 0.18, 0.36, 0.57, 0.71]

aspect_ratios_per_layer=[[0.5, 1.0, 2.0], #advance2
						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
						[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
						[0.5, 1.0, 2.0],
						[0.5, 1.0, 2.0]]
scales=[ 0.06, 0.16, 0.29, 0.43, 0.57,0.74]  #advance2

normalize_coords = True
ImageSize = (224,224)
limit_boxes = True
two_boxes_for_ar1 = True
coords = 'centroids'
variances=[0.1, 0.1, 0.2, 0.2]
classes = ["background","tumor"]

SegPath = r"E:\data_base\breast\weight_save\crop\fcn8_1024_Cross_4.h5"
#SSDPath = r"E:\data_base\breast\weight_save\crop\test_fcnsdd_model\fcn8_100epoch_1024_Cross_8_ssd_sdvance.h5"
SSDPath = r"E:\data_base\breast\weight_save\crop\fcn8_1024_Cross_4_ssd_sdvance2_max.h5"
#SSDPath = r"E:\data_base\breast\weight_save\crop\fcn8_100epoch_1024_Cross_9_ssd.h5"
model,predictor_sizes = FCN8_ssd_combine(Segpath=SegPath,
										 SSDpath = SSDPath,
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
                                predictor_sizes = predictor_sizes,
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
post = np.array([0,0,0,0,0])
post = post.reshape((1,5))
sa = list()
sa.append(post)
generator1 = BatchGenerator()
test_filenames,test_labels = generator1.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
												 annotations_paths=[r"E:\data_base\breast\xml_annotation"],
												 image_set_paths=[r"E:\data_base\breast\weight_save\crop\Cross_test_4.txt"],
												 #image_set_paths=[r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\test_test\test_one.txt"],
												 classes = classes,
												 ret=True)
test_generator = generator1.generate(
									 batch_size=1,
									 train=False,Transform=False,
									 ssd_box_encoder=ssd_box_encoder,
									 resize=(224,224))
import time
#存储结果
for k in range(len(test_filenames)):
	Img_Save_Path = r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\cross4\self_show_temp"
	Xml_Save_Path = r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\cross4\temp"
	Map_Save_Path = r"E:\data_base\breast\weight_save\crop\result\fcn_ssd\cross4\map"
	X, y_true, Row_X, Row_y, this_filenames = next(test_generator)  #X是resize的图片，y_true是预选框

	index = 0
	XX,name = os.path.split(this_filenames[index])
	ImgID = name.split(".")[0]

	Imgmap,y_pred  = model.predict(X)

	y_temp = np.copy(y_true)
	y_temp[:,:,0:-12]=y_pred       #用预测值替换实际值

	#解码得到真实坐标  y_pred_decoded是list类型，
	y_pred_decoded = decode_y(y_temp,
							  confidence_thresh=0.01,		#用于预选框筛选，小于confidence_thresh被去掉
							  iou_threshold=0.03,          #用于非极大值抑制,大于iou_threshold的被去掉
							  top_k=100,                     #设置"all"好一些？
							  input_coords='centroids',
							  normalize_coords=normalize_coords,  #从归一化坐标恢复
							  img_height=Row_X[index].shape[0],
							  img_width=Row_X[index].shape[1])
	y_pred_decoded = y_pred_decoded[index]   #
	#根据概率图得到目标
	pre = np.reshape(Imgmap[index],(224,224))*255.0
	#存储概率图
	img = Image.fromarray(pre.astype(np.uint8))
	img = img.resize((Row_X[index].shape[1],Row_X[index].shape[0]))  #恢复到原图大小
	img.save(os.path.join(Map_Save_Path,ImgID+".jpg"))
	img = np.array(img)
	pred_box = clcbox_from_map_all(np.array(img),marjin=10)    		 #由map图得到box
	#计算box和置信度
	box_class = list()
	#top_K = 10  #前10选均值
	IOU_thread = 0.2
	recall_thread = 0.35
	#根据pred_box，找到最匹配的y_pred_decoded
	for box in pred_box:
		similarities = iou(y_pred_decoded[:,-4:], box, coords='minmax')   #相似度
		recall_similarities = iou_recall(y_pred_decoded[:,-4:], box, coords='minmax')   #召回率
		sort_index = np.argsort(similarities)   #
		sort_index =sort_index[-1::-1]
		match_box = list()
		for index_sort in sort_index:
			if similarities[index_sort]>IOU_thread or recall_similarities[index_sort]>recall_thread:
				match_box.append(y_pred_decoded[index_sort,1])
		if len(match_box)>1:
			match_box = match_box[0:1]
		temp = np.zeros(pred_box[0].shape[0]+2)
		if len(match_box)==0:
			temp[0] = 1
			temp[1] = 0.01
		else:
			temp[0] = 1
			temp[1] = np.mean(np.array(match_box))
		temp[-4:] = box
		box_class.append(temp)

	#显示box_class，并存储
	plt.figure(1)
	time.sleep(0.01)
	plt.imshow(Row_X[index])
	time.sleep(0.01)
	current_axis = plt.gca()

	y_Row = Row_y[index]
	for y_true_i in y_Row:
		label = '{}'.format(classes[int(y_true_i[0])])
		current_axis.add_patch(plt.Rectangle((y_true_i[1], y_true_i[3]), y_true_i[2]-y_true_i[1], y_true_i[4]-y_true_i[3], color='red', fill=False, linewidth=2))
		current_axis.text(y_true_i[1], y_true_i[3], label, size='x-small', color='white', bbox={'facecolor':'red', 'alpha':1.0})

	for box in box_class:
		label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
		current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))
		current_axis.text(box[2], box[4], label, size='x-small', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
	plt.savefig(os.path.join(Img_Save_Path,ImgID+".jpg"))
	plt.clf()
	#存储xml文件
	for k in range(len(box_class)):
		box_class[k] = box_class[k][1:]
	write_box_to_xml(box_class,Xml_Save_Path,filename=ImgID+".xml",width=Row_X[index].shape[1],height=Row_X[index].shape[0],depth=3,isScore=True)
	print(ImgID+" has been written!")
print("over")

