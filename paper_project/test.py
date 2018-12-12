from bs4 import BeautifulSoup
from ssd_box_encode_decode_utils import *

# import tensorflow as tf
# import keras.backend as K
# from keras import metrics

# import shutil
# testpart =  r"E:\data_base\breast\test_part"
# deletepart = r"E:\data_base\breast\Analysis\to_be_deleted\3"
#
# deletelist  =os.listdir(deletepart)
# for file in deletelist:
# 	if os.path.exists(os.path.join(testpart,file)):
# 		os.remove(os.path.join(testpart,file))

# ypred = tf.Variable(np.random.random((5,4,4,1)),dtype=tf.float32)
# ytrue = tf.Variable(np.random.randint(0,2,(5,4,4,1)),dtype=tf.float32)
# loss2_4(ytrue, ypred)
# sa = focal_loss(y_true=ytrue,y_pred=ypred)

main_path = r"E:\data_base\breast\main_2000_jpg"
target_path = r"E:\data_base\breast\map_save\fcn8_ssd\compare_with_ssd300"
golden_path = r"E:\data_base\breast\xml_annotation"
xmlsit = [r"E:\data_base\breast\map_save\fcn8_ssd\xml",
		  r"E:\data_base\breast\map_save\ssd300\xml"]
paint_box_from_xml(main_path,golden_path,target_path,xmlsit=xmlsit)

print("over!")