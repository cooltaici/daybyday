from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from fcn_inception import *
from FCN4_inception_ssd import *
def combine_fcn_ssd(weightpath=r"",
					isnorm = True,
					image_size = (224,224),
					n_classes = 2,
					min_scale=None,
					max_scale=None,
					scale = [ 0.1, 0.23, 0.46, 0.61],
					aspect_ratios_global=None,
					aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
											[1.0/3.0, 0.5, 1.0, 2.0, 3.0],
											[0.5, 1.0, 2.0]],
					two_boxes_for_ar1=True,
					limit_boxes=False,
					variances=[0.1, 0.1, 0.2, 0.2],
					coords='centroids',
					normalize_coords=False):

	model,predictor_sizes = fcn4_inception3_ssd(weightpath=r"",
												isnorm = isnorm,
												image_size = image_size,
												n_classes = n_classes,
												min_scale=min_scale,
												max_scale=max_scale,
												scale = scale,
												aspect_ratios_global=aspect_ratios_global,
												aspect_ratios_per_layer=aspect_ratios_per_layer,
												two_boxes_for_ar1=True,
												limit_boxes=two_boxes_for_ar1,
												variances=variances,
												coords=coords,
												normalize_coords=False)

	if os.path.exists(weightpath):
		model.load_weights(weightpath)
	else:
		raise ValueError("路径不存在，无法加载")

	Seg_output = model.get_layer('Seg_output').output
	mbox_conf_softmax = model.get_layer('mbox_conf_softmax').output
	Img_input = model.input

	new_model = Model(inputs=Img_input,outputs=[Seg_output,mbox_conf_softmax])

	return new_model,predictor_sizes

#测试输出没问题
if __name__ == '__main__':
	model = fcn4_inception3_ssd(False)
	model.summary()
	print('over!')