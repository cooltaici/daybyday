import os
import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
###########################################################################
#文档说明：该图像生成器只是为了做肿瘤图像的分割，不需要用到annotations的信息
###########################################################################

#随机改变图像的亮度
def _brightness(image, min=0.5, max=2.0):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min,max)
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
#均衡化
def histogram_eq(image):
    image1 = np.copy(image)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1
#伽马变换
def gama_transform(image):  #小于1的时候是变亮，大于1的时候是变暗
    choice = np.random.random()
    if choice>0.7:
       sed = np.random.random()*0.7
       return exposure.adjust_gamma(image, sed)
    if choice<=0.7 and choice>=0.2:
       sed = np.random.random()*0.8+0.7
       return exposure.adjust_gamma(image, sed)
    if choice<=0.2:
       sed = np.random.random()*0.8+1.5
       return exposure.adjust_gamma(image, sed)

#以下预处理为修改keras自带
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

#图像生成类
class BatchGenerator:
    def __init__(self, box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'], filenames=None, labels=None):
        self.images_paths = None       #原图像路径
        self.mask_paths = None         #Mask图像路径
        self.image_set_path = None     #train or test filename

        if not filenames is None:
            if isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    self.Img_filenames = pickle.load(f)
            elif isinstance(filenames, (list, tuple)):
                self.Img_filenames = filenames
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.Img_filenames = [] # All unique image filenames will go here

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.Mask_filenames = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.Mask_filenames = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.Mask_filenames = [] # Each entry here will contain a 2D Numpy array with all the ground truth boxes for a given image


     #解析XML注释文件，获得文件名和标签
    def parse_xml(self,
                  images_paths = None,
                  mask_paths = None,
                  image_set_paths = None,
                  ret = False):
        self.images_paths = images_paths
        self.mask_paths = mask_paths
        self.image_set_paths = image_set_paths

        self.Img_filenames = []
        self.Mask_filenames = []

        #可能给的不止一个训练集
        for image_path, mask_path, image_set_path in zip(self.images_paths, self.mask_paths, self.image_set_paths):
            # 获得图像的ID
            with open(image_set_path) as f:
                image_ids = [line.strip() for line in f]   #默认移除字符串的首尾空格
            for image_id in image_ids:
                #判断在image_path和mask_path里面有没有相应的图像，二者都有的化，加入到训练集里去(绝对路径）
                temp_image_path = os.path.join(image_path,image_id + '.jpg')
                temp_mask_path = os.path.join(mask_path,image_id + '.bmp')
                if os.path.exists(temp_image_path) and os.path.exists(temp_mask_path):
                    self.Img_filenames.append(temp_image_path)
                    self.Mask_filenames.append(temp_mask_path)
        if ret:
            return self.Img_filenames, self.Mask_filenames


    #标准化以及预处理函数
    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= np.std(x, keepdims=True) + 1e-7
        return x

    #随机仿射变换相关
    def aplly_transform(self,x,y):
        if len(y.shape)==2:
            y = y.reshape((y.shape[0],y.shape[1],1))
        #tensorflow （batch,row,col,chanel)
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        if self.seed is not None:
            np.random.seed(self.seed)
        # use composition of homographies
        #旋转
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        #平移
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0
        #剪切
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        #缩放
        if self.zoom_range:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        else:
            zx, zy = 1, 1
        #转移矩阵
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
        #翻转
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                y = flip_axis(y, img_col_axis)
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                y = flip_axis(y, img_row_axis)
        # if len(y.shape)==3:
        #     y = y.reshape((y.shape[0], y.shape[1]))
        return (x,y)

    #yelid产生批量图像
    def generate(self,
                 batch_size=32,
                 train=True,           #是训练过程还是验证过程
                 Transform=True,         #是否打乱数据以及作变换
                 equalize=None,        #均衡化
                 brightness=False,     #亮度调整
                 gama = False,         #伽马变换
                 samplewise_center=False,              #中心化
                 samplewise_std_normalization=False,   #中心标准化
                 rotation_range=0.,      #旋转
                 width_shift_range=0.,   #水平平移
                 height_shift_range=0.,  #垂直平移
                 shear_range=0.,         #剪切
                 zoom_range=0.,           #缩放
                 horizontal_flip=False,  #水平翻转
                 vertical_flip=False,    #垂直翻转
                 resize=(224,224),
                 gray=False,
                 fill_mode = 'nearest',         #('constant', 'nearest', 'reflect' or 'wrap')
                 cval = 0,              #fill_mode为"constant"时有效
                 preprocessing_function=None,   #预处理函数
                 data_format = 'channels_last',
                 seed = None):
        #类成员赋值
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        self.batch_size = batch_size
        self.train = train
        self.Transform = Transform
        self.equalize = equalize
        self.gama = gama
        self.brightness = brightness
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.resize = resize
        self.gray = gray
        self.fill_mode = fill_mode
        self.cval = cval
        self.preprocessing_function = preprocessing_function
        self.seed = seed
        #是否打乱数据
        if self.seed is not None:
            np.random.seed(seed)
        if train:
            if self.Transform:
                self.Img_filenames, self.Mask_filenames = shuffle(self.Img_filenames, self.Mask_filenames) #打乱数据
        current = 0  #从样本0开始

        while True:
            batch_X, batch_Y = [], []
            #超过一个epoch就打乱数据,但如果尺训练过程，就不打乱
            if current >= len(self.Img_filenames):
                if train:
                    if self.Transform:
                        self.Img_filenames, self.Mask_filenames = shuffle(self.Img_filenames, self.Mask_filenames)
                current = 0
            #读取原图像
            for filename in self.Img_filenames[current:current+batch_size]:
                if not os.path.exists(filename):
                    raise ValueError(filename + " is not exists")
                with Image.open(filename) as img:
                    batch_X.append(np.array(img))
            #读取mask图像：二值化图像。由于每次都是直接根据文件名读取的图像，所以不需要深度拷贝
            for filename in self.Mask_filenames[current:current+self.batch_size]:
                if not os.path.exists(filename):
                    raise ValueError(filename+ " is not exists")
                with Image.open(filename) as img:
                    img = img.convert("L")
                    batch_Y.append(np.array(img))
            this_filenames = self.Img_filenames[current:current+self.batch_size]  #当前batch的样本名
            current += self.batch_size

            if self.train:
                #下面对每一幅图片和相应的标签进行变换处理
                #print(this_filenames)
                if self.Transform:
                    for i in range(len(batch_X)):
                        img_height, img_width, ch = batch_X[i].shape
                        batch_X[i] = batch_X[i]/255.0
                        batch_Y[i][batch_Y[i]<127] = 0
                        batch_Y[i][batch_Y[i] >=127] = 1
                        #固定输入大小
                        if resize:
                            batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                            batch_Y[i] = cv2.resize(batch_Y[i], dsize=(resize[1], resize[0]))
                        #伽马变换
                        if self.gama:
                            batch_X[i] = gama_transform(batch_X[i])
                        #随机直方图均衡化
                        if self.equalize:
                            p = np.random.uniform(0, 1)
                            if p<equalize:
                                batch_X[i] = histogram_eq(batch_X[i])
                        #随机白化
                        if self.brightness:
                            p = np.random.uniform(0,1)
                            if p < brightness[2]:
                                batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])
                        #仿射变换，都要改变
                        # for index in range(len(batch_X)):
                        #     if batch_X[index] is None:
                        #         print("Index:"+str(i)+"错误！"+":"+this_filenames[index])
                        (batch_X[i],batch_Y[i]) = self.aplly_transform(batch_X[i],batch_Y[i])
                        if len(batch_Y[i].shape)==2:
                            batch_Y[i] = batch_Y[i].reshape((batch_Y[i].shape[0],batch_Y[i].shape[1],1))
                        # batch_X[i] = tempx.copy()
                        # batch_Y[i] = tempy.copy()
                        #输入标准化: 是否使用还需要考查，没达到理想效果
                        #batch_X[i] = self.standardize(batch_X[i])
                        if gray:
                            batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), axis=2)
                    yield (np.array(batch_X), np.array(batch_Y))
                else:
                    for i in range(len(batch_X)):
                        batch_X[i] = batch_X[i]/255.0
                        batch_Y[i][batch_Y[i]<127] = 0
                        batch_Y[i][batch_Y[i] >=127] = 1
                        if resize:
                            batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                            batch_Y[i] = cv2.resize(batch_Y[i], dsize=(resize[1], resize[0]))
                        if len(batch_Y[i].shape)==2:
                            batch_Y[i] = batch_Y[i].reshape((batch_Y[i].shape[0],batch_Y[i].shape[1],1))
                        if gray:
                            batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), axis=2)
                    yield (np.array(batch_X), np.array(batch_Y))
            else:
                original_images = np.copy(batch_X)    # The original, unaltered images
                original_labels = deepcopy(batch_Y)    # The original, unaltered labels
                for i in range(len(batch_X)):
                    batch_X[i] = batch_X[i] / 255.0
                    batch_Y[i][batch_Y[i] < 127] = 0
                    batch_Y[i][batch_Y[i] >= 127] = 1
                    if resize:
                        batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                        batch_Y[i] = cv2.resize(batch_Y[i], dsize=(resize[1], resize[0]))
                    if len(batch_Y[i].shape) == 2:
                        batch_Y[i] = batch_Y[i].reshape((batch_Y[i].shape[0], batch_Y[i].shape[1], 1))
                    if gray:
                        batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), axis=2)
                yield (np.array(batch_X), np.array(batch_Y), original_images, original_labels,this_filenames)

    #存储文件名和标签,以pickle的方式
    def save_filenames_and_labels(self, filenames_path='filenames.pkl', labels_path='labels.pkl'):
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(self.labels, f)

if __name__=='__main__':
    generator0 = BatchGenerator()
    train_filenames,train_Mask_filenames = generator0.parse_xml(images_paths=[r"E:\data_base\breast\main_2000_jpg"],
                                                            mask_paths=[r"E:\data_base\breast\main_2000_mask_bmp"],
                                                            image_set_paths=[r"E:\data_base\breast\train.txt"],ret = True)
    train_generator = generator0.generate(
                                         batch_size=1,
                                         #brightness = [0.5,0.7,0.5],
                                         resize = [224, 224],gama=True,
                                         zoom_range=[0.9,1.20],
                                         rotation_range = 18,
                                         width_shift_range = 0.12,
                                         height_shift_range = 0.12,
                                         horizontal_flip = True,
                                         train = True)
    for sa in train_generator:
        print(sa[0].shape)
        print(sa[1].shape)
    print('over!')