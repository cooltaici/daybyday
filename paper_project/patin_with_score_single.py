import numpy as np
import cv2
from bs4 import BeautifulSoup
import os
from PIL import Image
import matplotlib.pyplot as plt
from ssd_box_encode_decode_utils import *
import time

main_path = r"E:\data_base\breast\main_2000_jpg"
golden_path = r"E:\data_base\breast\xml_annotation"
fcn8_ssd_path = r"E:\data_base\breast\map_save\fcn8_ssd\xml"
ssd300_path = r"E:\data_base\breast\map_save\ssd300\xml"
target_path = r"E:\data_base\breast\map_save\fcn8_ssd\compare_with_ssd300\single"
sample_list = os.listdir(fcn8_ssd_path)
sample_list.sort()

for filename in sample_list:
	ImgID = filename.split(".")[0]
	if not os.path.exists(os.path.join(main_path,ImgID+".jpg")):
		continue
	if not os.path.exists(os.path.join(golden_path,filename)):
		continue
	if not os.path.exists(os.path.join(ssd300_path,filename)):
		continue
	Img = np.array(Image.open(os.path.join(main_path,ImgID+".jpg")))
	golden_boxes = get_box_from_xml(os.path.join(golden_path,filename), isScore=False)
	ssd300_boxes = get_box_from_xml(os.path.join(ssd300_path,filename),isScore=True)
	fcn8_ssd_boxes = get_box_from_xml(os.path.join(fcn8_ssd_path,filename),isScore=True)

	box_fcn = 0
	score_max = 0
	for box in fcn8_ssd_boxes:
		if box[0]>score_max:
			box_fcn = box
			score_max = box[0]
	box_ssd = 0
	score_max = 0
	for box in ssd300_boxes:
		if box[0]>score_max:
			box_ssd = box
			score_max = box[0]

	plt.figure(1)
	time.sleep(0.01)
	plt.imshow(Img)
	time.sleep(0.01)
	current_axis = plt.gca()

	#显示金标准
	for box in golden_boxes:
		current_axis.add_patch(plt.Rectangle((box[-4], box[-2]), box[-3]-box[-4], box[-1]-box[-2], color='red', fill=False, linewidth=2))
	#
	label = '{:.2f}'.format(box_fcn[0])
	current_axis.add_patch(plt.Rectangle((box_fcn[-4], box_fcn[-2]), box_fcn[-3]-box_fcn[-4], box_fcn[-1]-box_fcn[-2], color='blue', fill=False, linewidth=2))
	current_axis.text(box_fcn[-4], box_fcn[-2], label, size='x-small', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
	#
	label = '{:.2f}'.format(box_ssd[0])
	current_axis.add_patch(plt.Rectangle((box_ssd[-4], box_ssd[-2]), box_ssd[-3]-box_ssd[-4], box_ssd[-1]-box_ssd[-2], color='green', fill=False, linewidth=2))
	current_axis.text(box_ssd[-4], box_ssd[-2], label, size='x-small', color='white', bbox={'facecolor':'green', 'alpha':1.0})

	plt.savefig(os.path.join(target_path,ImgID+".jpg"))
	plt.clf()
	print(ImgID+"has been saved!")
