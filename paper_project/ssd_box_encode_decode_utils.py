import numpy as np
import cv2
from bs4 import BeautifulSoup
import os
from PIL import Image
import matplotlib.pyplot as plt

#从给出的xml文件中，获取box,返回list(list(xmin,xmax,ymin,ymax))（已验证）
def get_box_from_xml(path, isScore=False):
    with open(path) as f:
        soup = BeautifulSoup(f, 'xml')
    objects = soup.find_all('object')
    box_list = list()
    for obj in objects:
        xmin = int(float(obj.bndbox.xmin.text.strip()))
        ymin = int(float(obj.bndbox.ymin.text.strip()))
        xmax = int(float(obj.bndbox.xmax.text.strip()))
        ymax = int(float(obj.bndbox.ymax.text.strip()))
        if isScore:
            try:
                score = float(obj.bndbox.score.text.strip())
                box_list.append([score,xmin,xmax,ymin,ymax])
            except:
                box_list.append([xmin,xmax,ymin,ymax])
        else:
            box_list.append([xmin,xmax,ymin,ymax])
    #width = int(soup.size.width.text)
    #height = int(soup.size.height.text)
    return box_list

#把xml文件中的box信息画到原图中,（已验证）
'''
入参：
    mainpath： 原图片位置
    golden_xml：金标准xml文件
    target_path： 图片目标存储位置
    xmlsit： 其它xml文件位置
'''
def paint_box_from_xml(mainpath,golden_xml,target_path,xmlsit=[]):
    if not os.path.exists(mainpath):
        raise ValueError("mainpath is not exist")
    colorbar = [[255,255,255],[255,0,0],[0,255,0],[0,0,255]]   #分别是白，蓝，绿，红
    line_width = 2
    for path in xmlsit:
        if not os.path.exists(path):
            raise ValueError(path+": is wrong!")
    if xmlsit:
        firstlist = os.listdir(xmlsit[0])
        for sample in firstlist:
            ImgID = sample.split(".")[0]
            Img = Image.open(os.path.join(mainpath,ImgID+".jpg"))
            Img = np.array(Img)
            #先画Goldenl框
            box_list = get_box_from_xml(os.path.join(golden_xml,sample))
            for box in box_list:
                cv2.rectangle(Img,(box[0],box[2]),(box[1],box[3]),colorbar[0],line_width)
                #cv2.addText(Img,"Ground",(box[0],box[2]-10),"Times",color=colorbar[0],pointSize=10)
            #再画其它框
            for kk in range(len(xmlsit)):
                box_list = get_box_from_xml(os.path.join(xmlsit[kk],sample))
                for box in box_list:
                    cv2.rectangle(Img,(box[0],box[2]),(box[1],box[3]),colorbar[kk+1],line_width)
                    #cv2.addText(Img,"Ground",(box[0],box[2]-10),"Times",color=colorbar[kk],pointSize=10)
            cv2.imwrite(os.path.join(target_path,ImgID+".jpg"),Img)
            #Img = Image.fromarray(Img)
            #Img.save()
            print(ImgID+" has been written！")


#把box框存储为为XML文件
"""
boxlist:  list(np.array)类型，表示一个样本的所有检测框
targetpath： 存储xml文件的目标路径
filename: 存储的文件名
width:
height:
depth:
isScore: 是否存储分数
"""
def write_box_to_xml(boxlist,targetpath,filename=r"2018_100000.xml",width=765,height=600,depth=3,isScore = False):
    if boxlist:
        #基本信息
        ImgID = filename.split(".")[0]
        soup = BeautifulSoup("<annotation><folder>main_2000_jpg</folder></annotation>","xml")
        #文件名
        newTag = soup.new_tag("filename")
        newTag.string = ImgID+".jpg"
        soup.annotation.append(newTag)
        #size
        sizeTag = soup.new_tag("size")
        newTag = soup.new_tag("width")
        newTag.string = str(width)
        sizeTag.append(newTag)
        newTag = soup.new_tag("height")
        newTag.string = str(height)
        sizeTag.append(newTag)
        newTag = soup.new_tag("depth")
        newTag.string = str(depth)
        sizeTag.append(newTag)
        soup.annotation.append(sizeTag)
        #Segment
        newTag = soup.new_tag("Segmented")
        newTag.string = str(1)
        soup.annotation.append(sizeTag)
        for box in boxlist:
            objectTag = soup.new_tag("object")

            newTag = soup.new_tag("name")
            newTag.string = "tumor"
            objectTag.append(newTag)

            newTag = soup.new_tag("difficult")
            newTag.string = "0"
            objectTag.append(newTag)

            bndboxTag = soup.new_tag("bndbox")
            if isScore:   #加上分数
                newTag = soup.new_tag("score")
                newTag.string = str(box[-5])
                bndboxTag.append(newTag)
            newTag = soup.new_tag("xmin")
            newTag.string = str(box[-4])
            bndboxTag.append(newTag)
            newTag = soup.new_tag("xmax")
            newTag.string = str(box[-3])
            bndboxTag.append(newTag)
            newTag = soup.new_tag("ymin")
            newTag.string = str(box[-2])
            bndboxTag.append(newTag)
            newTag = soup.new_tag("ymax")
            newTag.string = str(box[-1])
            bndboxTag.append(newTag)
            objectTag.append(bndboxTag)

            soup.annotation.append(objectTag)
        #存储
        if os.path.exists(targetpath):
            open(os.path.join(targetpath,ImgID+".xml"),"w").write(soup.prettify())
    else:
        raise TabError("boxlist are NULL")

#只选取一个最大的BOX，返回的是 np.array([xmin,xmax,ymin,ymax])
def clcbox_from_map(Probgraph,marjin=5):
    thre = 120   #这个是获取ROI区域，可以设置小一点
    newI = np.array([])
    for k in range(20):
        _, newI = cv2.threshold(Probgraph, thre, 255, cv2.THRESH_BINARY)
        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
        se2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        newI = cv2.dilate(newI, se2)
        newI = cv2.erode(newI, se1)
        if np.sum(newI.astype(np.uint8))==0:
            thre = thre - 10
            if thre < 45:
                return 0,0,0,0
        else:
            break
    _, newI = cv2.threshold(Probgraph, thre, 255, cv2.THRESH_BINARY)

    se2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marjin,marjin))
    newI = cv2.dilate(newI, se2)  #膨胀
    #选择最大连通区域
    _, contours,_ = cv2.findContours(newI,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    maxroi = 0
    index = -1
    for kk in range(len(contours)):
        listcontour = list()
        listcontour.append(contours[kk])
        tmp = np.zeros((Probgraph.shape[0],Probgraph.shape[1]))
        tmp = cv2.drawContours(tmp, listcontour, -1, [255, 255, 255], thickness=-1)
        if np.sum(tmp)>maxroi:
            maxroi = np.sum(tmp)
            index = kk
    tmplist = list()
    tmplist.append(contours[index])
    tmp = np.zeros((Probgraph.shape[0],Probgraph.shape[1]))
    tmp = cv2.drawContours(tmp, tmplist, -1, [255, 255, 255], thickness=-1)
    #计算外矩形
    x,y,w,h = cv2.boundingRect(tmp.astype(np.uint8))
    #ROI = (x1,y1,w1,h1)
    #cropImg = im[y:y+h,x:x+w]
    # cropMask = mask[y:y+h,x:x+w]
    return np.array([x,x+w,y,y+h])

#选取所有符合的BOX，返回的是list(np.array([xmin,xmax,ymin,ymax]))
def clcbox_from_map_all(Probgraph,marjin=5):
    thre = 200   #这个是获取ROI区域，可以设置小一点
    newI = np.array([])
    for k in range(20):
        _, newI = cv2.threshold(Probgraph, thre, 255, cv2.THRESH_BINARY)
        # se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))  #先前使用的
        # se2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))    ##先前使用的
        se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
        se2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        newI = cv2.dilate(newI, se2)
        newI = cv2.erode(newI, se1)
        if np.sum(newI.astype(np.uint8))<=100*255:             #按阈值决定是否继续减小thre
            thre = thre - 10
            if thre < 50:
                return [np.array([0,0,0,0])]
        else:
            break
    _, newI = cv2.threshold(Probgraph, thre, 255, cv2.THRESH_BINARY)
    #连接比较靠近的区域
    se1= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(marjin*1.5),int(marjin*1.5)))
    newI = cv2.dilate(newI, se1)
    newI = cv2.erode(newI, se1)
    #去掉小的散点
    se1= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(marjin*0.9),int(marjin*0.9)))
    newI = cv2.erode(newI, se1)
    se1= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(marjin*2.1),int(marjin*2.1)))
    newI = cv2.dilate(newI, se1)  #膨胀
    newI = newI.astype(np.uint8)
    #找出所有边界
    _, contours,_ = cv2.findContours(newI,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    box_all = list()
    for kk in range(len(contours)):
        listcontour = list()
        listcontour.append(contours[kk])
        tmp = np.zeros((Probgraph.shape[0],Probgraph.shape[1]))
        tmp = cv2.drawContours(tmp, listcontour, -1, [255, 255, 255], thickness=-1)
        x,y,w,h = cv2.boundingRect(tmp.astype(np.uint8))
        box_all.append(np.array([x,x+w,y,y+h]))
    return box_all

def iou(boxes1, boxes2, coords='centroids'):
    if len(boxes1.shape) > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    if coords == 'centroids':
        # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:,2]))
    union = (boxes1[:,1] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,2]) + (boxes2[:,1] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,2]) - intersection

    return intersection / union

def iou_recall(boxes1, boxes2, coords='centroids'):
    if len(boxes1.shape) > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    if coords == 'centroids':
        # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:,2]))
    recall = (boxes2[:,1]-boxes2[:,0])*(boxes2[:,3] - boxes2[:,2])
    #union = (boxes1[:,1] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,2]) + (boxes2[:,1] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,2]) - intersection

    return intersection / recall

def convert_coordinates(tensor, start_index, conversion='minmax2centroids'):
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1

def _greedy_nms(predictions, iou_threshold=0.45, coords='minmax'):
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # 计算概率最大的box
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # 移除重合率大的box
    return np.array(maxima)

#修改
def decode_y(y_pred,
             confidence_thresh=0.01,
             iou_threshold=0.45,
             top_k=200,
             input_coords='centroids',
             normalize_coords=False,
             img_height=None,
             img_width=None):
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,-4:-2] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,-2:] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='minmax') # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred:
            pred = np.concatenate(pred, axis=0)
        else:
            pred = np.array([0,0,0,0,0,0]).reshape((1,6))
        if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
            top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list
    return y_pred_decoded

class SSDBoxEncoder:
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=None,
                 max_scale=None,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.3,
                 coords='centroids',
                 normalize_coords=False):
        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != len(predictor_sizes)+1): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else: # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        if aspect_ratios_per_layer:
            if (len(aspect_ratios_per_layer) != len(predictor_sizes)): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                aspect_ratios = np.array(aspect_ratios)
                if np.any(aspect_ratios <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if not aspect_ratios_global:
                raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` cannot be `None`.")
            aspect_ratios_global = np.array(aspect_ratios_global)
            if np.any(aspect_ratios_global <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (coords == 'minmax' or coords == 'centroids'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if aspect_ratios_per_layer:
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

    def generate_anchor_boxes(self,
                              batch_size,
                              feature_map_size,
                              aspect_ratios,
                              this_scale,
                              next_scale,
                              diagnostics=False):
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        n_boxes = len(aspect_ratios)
        for ar in aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular anchor box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w,h))
                # Add 1 to `n_boxes` since we seem to have two boxes for aspect ratio 1
                n_boxes += 1
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width/2, self.img_width-cell_width/2, feature_map_size[1])
        cy = np.linspace(cell_height/2, self.img_height-cell_height/2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.limit_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 1]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 1]] = x_coords
            y_coords = boxes_tensor[:,:,:,[2, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[2, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, :2] /= self.img_width
            boxes_tensor[:, :, :, 2:] /= self.img_height

        if self.coords == 'centroids':
            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
            # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))

        # Now reshape the 5D tensor above into a 3D tensor of shape
        # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
        # order of the tensor content will be identical to the order obtained from the reshaping operation
        # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
        # use the same default index order, which is C-like index ordering)
        boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))

        if diagnostics:
            return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor

    def generate_encode_template(self, batch_size, diagnostics=False):

        # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
        #    If `scales` is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`
        if self.scales is None:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)

        # 2: For each conv predictor layer (i.e. for each scale factor) get the tensors for
        #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = [] # List to hold the box widths and heights
            cell_sizes = [] # List to hold horizontal and vertical distances between any two boxes
            if self.aspect_ratios_per_layer: # If individual aspect ratios are given per layer, we need to pass them to `generate_anchor_boxes()` accordingly
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
            else: # Use the same global aspect ratio list for all layers
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_global,
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
        else:
            if self.aspect_ratios_per_layer:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))
            else:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_global,
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))

        boxes_tensor = np.concatenate(boxes_tensor, axis=1) # Concatenate the anchor tensors from the individual layers to one

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, wh_list, cell_sizes
        else:
            return y_encode_template

    def encode_y(self, ground_truth_labels, isbox = False):

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

        class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]): # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[1])) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
            negative_boxes = np.ones((y_encode_template.shape[1])) # 1 for all negative boxes, 0 otherwise
            for true_box in ground_truth_labels[i]: # For each ground truth box belonging to the current batch item...
                true_box = true_box.astype(np.float)
                if abs(true_box[2] - true_box[1] < 0.001) or abs(true_box[4] - true_box[3] < 0.001): continue # Protect ourselves against bad ground truth data: boxes with width or height equal to zero
                if self.normalize_coords:
                    true_box[1:3] /= self.img_width # Normalize xmin and xmax to be within [0,1]
                    true_box[3:5] /= self.img_height # Normalize ymin and ymax to be within [0,1]
                if self.coords == 'centroids':
                    true_box = convert_coordinates(true_box, start_index=1, conversion='minmax2centroids')
                similarities = iou(y_encode_template[i,:,-12:-8], true_box[1:], coords=self.coords) # The iou similarities for all anchor boxes
                negative_boxes[similarities >= self.neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1
        if not isbox:
            y_encoded = y_encoded[:,:,:-12]   #只保留类别信息（background,tumor)
        return y_encoded
