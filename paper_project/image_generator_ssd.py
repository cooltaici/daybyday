
import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from copy import deepcopy
from PIL import Image
import csv
import os
from bs4 import BeautifulSoup
import pickle


def _translate(image, horizontal=(0,40), vertical=(0,10)):
    rows,cols,ch = image.shape
    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift

def _flip(image, orientation='horizontal'):
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)

def _scale(image, min=0.9, max=1.1):
    rows,cols,ch = image.shape
    scale = np.random.uniform(min, max)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale

def _brightness(image, min=0.5, max=2.0):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min,max)
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def histogram_eq(image):
    image1 = np.copy(image)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

class BatchGenerator:
    def __init__(self, box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax']):
        # These are the variables we always need
        self.include_classes = None
        self.box_output_format = box_output_format

        # These are the variables that we only need if we want to use parse_csv()
        self.image_path = None
        self.labels_path = None
        self.input_format = None

        # These are the variables that we only need if we want to use parse_xml()
        self.images_paths = None
        self.annotations_path = None
        self.image_set_path = None
        self.image_set = None
        self.classes = None
        self.filenames = [] # All unique image filenames will go here
        self.labels = [] # Each entry here will contain a 2D Numpy array with all the ground truth boxes for a given image

    def parse_xml(self,
                  images_paths=None,
                  annotations_paths=None,
                  image_set_paths=None,
                  classes=['background',
                           'tumor'],
                  ret=False):
        #返回self.filenames（样本名）, self.labels（样本队名的目标，不止一个）
        if not images_paths is None: self.images_paths = images_paths
        if not annotations_paths is None: self.annotations_paths = annotations_paths
        if not image_set_paths is None: self.image_set_paths = image_set_paths
        if not classes is None: self.classes = classes


        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        for image_path, image_set_path, annotations_path in zip(self.images_paths, self.image_set_paths, self.annotations_paths):
            # Parse the image set that so that we know all the IDs of all the images to be included in the dataset
            with open(image_set_path) as f:
                image_ids = [line.strip() for line in f]

            # Parse the labels for each image ID from its respective XML file
            for image_id in image_ids:
                # Open the XML file for this image
                with open(os.path.join(annotations_path, image_id+'.xml')) as f:
                    soup = BeautifulSoup(f, 'xml')

                folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                filename = soup.filename.text
                self.filenames.append(os.path.join(image_path, filename))

                boxes = [] # We'll store all boxes for this image here
                objects = soup.find_all('object') # Get a list of all objects in this image

                # Parse the data for each object
                for obj in objects:
                    class_name = obj.find('name').text
                    class_id = self.classes.index(class_name)
                    # Check if this class is supposed to be included in the dataset
                    xmin = int(obj.bndbox.xmin.text)
                    ymin = int(obj.bndbox.ymin.text)
                    xmax = int(obj.bndbox.xmax.text)
                    ymax = int(obj.bndbox.ymax.text)
                    item_dict = {'folder': folder,
                                 'image_name': filename,
                                 'image_id': image_id,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    for item in self.box_output_format:
                        box.append(item_dict[item])
                    boxes.append(box)

                self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels


    def generate(self,
                 batch_size=32,
                 train=True,
                 Transform=True,         #是否打乱数据
                 ssd_box_encoder=None,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 max_crop_and_resize=False,
                 full_crop_and_resize=False,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 gray=False,
                 limit_boxes=True,
                 include_thresh=0.3):
        if train:
            if Transform:
                self.filenames, self.labels = shuffle(self.filenames, self.labels) # Shuffle the data before we begin
        current = 0

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        while True:
            batch_X, batch_y = [], []
            #Shuffle the data after each complete pass
            if current >= len(self.filenames):
                if Transform:                     #不做变换也就不会打乱数据
                    self.filenames, self.labels = shuffle(self.filenames, self.labels)
                current = 0
            for filename in self.filenames[current:current+batch_size]:
                with Image.open(filename) as img:
                    batch_X.append(np.array(img))
            batch_y = deepcopy(self.labels[current:current+batch_size])
            this_filenames = self.filenames[current:current+batch_size] # The filenames of the files in the current batch
            current += batch_size

            if train:
                if Transform:
                    # At this point we're done producing the batch. Now perform some
                    # optional image transformations:
                    batch_items_to_remove = [] # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list
                    for i in range(len(batch_X)):
                        img_height, img_width, ch = batch_X[i].shape


                        if equalize:
                            batch_X[i] = histogram_eq(batch_X[i])

                        if brightness:
                            p = np.random.uniform(0,1)
                            if p >= (1-brightness[2]):
                                batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])

                        # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks
                        batch_y[i] = np.array(batch_y[i])
                        batch_X[i] = batch_X[i]/255.0

                        if flip:
                            p = np.random.uniform(0,1)
                            if p >= (1-flip):
                                batch_X[i] = _flip(batch_X[i])
                                batch_y[i][:,[xmin,xmax]] = img_width - batch_y[i][:,[xmax,xmin]] # xmin and xmax are swapped when mirrored

                        if translate:
                            p = np.random.uniform(0,1)
                            if p >= (1-translate[2]):
                                # Translate the image and return the shift values so that we can adjust the labels
                                batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                                # Adjust the labels
                                batch_y[i][:,[xmin,xmax]] += xshift
                                batch_y[i][:,[ymin,ymax]] += yshift
                                # Limit the box coordinates to lie within the image boundaries
                                if limit_boxes:
                                    before_limiting = deepcopy(batch_y[i])
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                    # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                    # process that they don't serve as useful training examples anymore, because too little of them is
                                    # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                    # `include_thresh` of the box area before limiting.
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                    if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                        if scale:
                            p = np.random.uniform(0,1)
                            if p >= (1-scale[2]):
                                # Rescale the image and return the transformation matrix M so we can use it to adjust the box coordinates
                                batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                                # Adjust the box coordinates
                                # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                                toplefts = np.array([batch_y[i][:,xmin], batch_y[i][:,ymin], np.ones(batch_y[i].shape[0])])
                                bottomrights = np.array([batch_y[i][:,xmax], batch_y[i][:,ymax], np.ones(batch_y[i].shape[0])])
                                new_toplefts = (np.dot(M, toplefts)).T
                                new_bottomrights = (np.dot(M, bottomrights)).T
                                batch_y[i][:,[xmin,ymin]] = new_toplefts.astype(np.int)
                                batch_y[i][:,[xmax,ymax]] = new_bottomrights.astype(np.int)
                                # Limit the box coordinates to lie within the image boundaries
                                if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                                    before_limiting = deepcopy(batch_y[i])
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                    if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                        if max_crop_and_resize:
                            # The ratio of the two aspect ratios (source image and target size) determines the maximal possible crop.
                            image_aspect_ratio = img_width / img_height
                            resize_aspect_ratio = max_crop_and_resize[1] / max_crop_and_resize[0]

                            if image_aspect_ratio < resize_aspect_ratio:
                                crop_width = img_width
                                crop_height = int(round(crop_width / resize_aspect_ratio))
                            else:
                                crop_height = img_height
                                crop_width = int(round(crop_height * resize_aspect_ratio))
                            # The actual cropping and resizing will be done by the random crop and resizing operations below.
                            # Here, we only set the parameters for them.
                            random_crop = (crop_height, crop_width, max_crop_and_resize[2], max_crop_and_resize[3])
                            resize = (max_crop_and_resize[0], max_crop_and_resize[1])

                        if full_crop_and_resize:

                            resize_aspect_ratio = full_crop_and_resize[1] / full_crop_and_resize[0]

                            if img_width < img_height:
                                crop_height = img_height
                                crop_width = int(round(crop_height * resize_aspect_ratio))
                            else:
                                crop_width = img_width
                                crop_height = int(round(crop_width / resize_aspect_ratio))
                            # The actual cropping and resizing will be done by the random crop and resizing operations below.
                            # Here, we only set the parameters for them.
                            if max_crop_and_resize:
                                p = np.random.uniform(0,1)
                                if p >= (1-full_crop_and_resize[4]):
                                    random_crop = (crop_height, crop_width, full_crop_and_resize[2], full_crop_and_resize[3])
                                    resize = (full_crop_and_resize[0], full_crop_and_resize[1])

                        if random_crop:
                            y_range = img_height - random_crop[0]
                            x_range = img_width - random_crop[1]
                            # Keep track of the number of trials and of whether or not the most recent crop contains at least one object
                            min_1_object_fulfilled = False
                            trial_counter = 0
                            while (not min_1_object_fulfilled) and (trial_counter < random_crop[3]):
                                # Select a random crop position from the possible crop positions
                                if y_range >= 0: crop_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension
                                else: crop_ymin = np.random.randint(0, -y_range + 1) # The possible positions for the image on the background canvas in the vertical dimension
                                if x_range >= 0: crop_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension
                                else: crop_xmin = np.random.randint(0, -x_range + 1) # The possible positions for the image on the background canvas in the horizontal dimension
                                # Perform the crop
                                if y_range >= 0 and x_range >= 0: # If the patch to be cropped out is smaller than the original image in both dimenstions, we just perform a regular crop
                                    # Crop the image
                                    patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0], crop_xmin:crop_xmin+random_crop[1]])
                                    # Translate the box coordinates into the new coordinate system: Cropping shifts the origin by `(crop_ymin, crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] -= crop_ymin
                                    patch_y[:,[xmin,xmax]] -= crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Both the x- and y-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        y_coords = patch_y[:,[ymin,ymax]]
                                        y_coords[y_coords < 0] = 0
                                        y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                        patch_y[:,[ymin,ymax]] = y_coords
                                        x_coords = patch_y[:,[xmin,xmax]]
                                        x_coords[x_coords < 0] = 0
                                        x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                        patch_y[:,[xmin,xmax]] = x_coords
                                elif y_range >= 0 and x_range < 0: # If the crop is larger than the original image in the horizontal dimension only,...
                                    # Crop the image
                                    patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0]]) # ...crop the vertical dimension just as before,...
                                    canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                    canvas[:, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                                    patch_X = canvas
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(crop_ymin, -crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] -= crop_ymin
                                    patch_y[:,[xmin,xmax]] += crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Only the y-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        y_coords = patch_y[:,[ymin,ymax]]
                                        y_coords[y_coords < 0] = 0
                                        y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                        patch_y[:,[ymin,ymax]] = y_coords
                                elif y_range < 0 and x_range >= 0: # If the crop is larger than the original image in the vertical dimension only,...
                                    # Crop the image
                                    patch_X = np.copy(batch_X[i][:,crop_xmin:crop_xmin+random_crop[1]]) # ...crop the horizontal dimension just as in the first case,...
                                    canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                    canvas[crop_ymin:crop_ymin+img_height, :] = patch_X # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                                    patch_X = canvas
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] += crop_ymin
                                    patch_y[:,[xmin,xmax]] -= crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Only the x-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        x_coords = patch_y[:,[xmin,xmax]]
                                        x_coords[x_coords < 0] = 0
                                        x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                        patch_y[:,[xmin,xmax]] = x_coords
                                else:  # If the crop is larger than the original image in both dimensions,...
                                    patch_X = np.copy(batch_X[i])
                                    canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                    canvas[crop_ymin:crop_ymin+img_height, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                                    patch_X = canvas
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, -crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] += crop_ymin
                                    patch_y[:,[xmin,xmax]] += crop_xmin
                                if limit_boxes and (y_range >= 0 or x_range >= 0):
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (patch_y[:,xmax] - patch_y[:,xmin]) * (patch_y[:,ymax] - patch_y[:,ymin])
                                    if include_thresh == 0: patch_y = patch_y[after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: patch_y = patch_y[after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all
                                trial_counter += 1 # We've just used one of our trials
                                # Check if we have found a valid crop
                                if random_crop[2] == 0: # If `min_1_object == 0`, break out of the while loop after the first loop because we are fine with whatever crop we got
                                    batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                    batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                    # Update the image size so that subsequent transformations can work correctly
                                    img_height = random_crop[0]
                                    img_width = random_crop[1]
                                    break
                                elif len(patch_y) > 0: # If we have at least one object left, this crop is valid and we can stop
                                    min_1_object_fulfilled = True
                                    batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                    batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                    # Update the image size so that subsequent transformations can work correctly
                                    img_height = random_crop[0]
                                    img_width = random_crop[1]
                                elif (trial_counter >= random_crop[3]) and (not i in batch_items_to_remove): # If we've reached the trial limit and still not found a valid crop, remove this image from the batch
                                    batch_items_to_remove.append(i)

                        if crop:
                            # Crop the image
                            batch_X[i] = np.copy(batch_X[i][crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])
                            # Translate the box coordinates into the new coordinate system if necessary: The origin is shifted by `(crop[0], crop[2])` (i.e. by the top and left crop values)
                            # If nothing was cropped off from the top or left of the image, the coordinate system stays the same as before
                            if crop[0] > 0:
                                batch_y[i][:,[ymin,ymax]] -= crop[0]
                            if crop[2] > 0:
                                batch_y[i][:,[xmin,xmax]] -= crop[2]
                            # Update the image size so that subsequent transformations can work correctly
                            img_height -= crop[0] + crop[1]
                            img_width -= crop[2] + crop[3]
                            # Limit the box coordinates to lie within the new image boundaries
                            if limit_boxes:
                                before_limiting = np.copy(batch_y[i])
                                # We only need to check those box coordinates that could possibly have been affected by the cropping
                                # For example, if we only crop off the top and/or bottom of the image, there is no need to check the x-coordinates
                                if crop[0] > 0:
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                if crop[1] > 0:
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                if crop[2] > 0:
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                if crop[3] > 0:
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                        if resize:
                            batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                            batch_y[i][:,[xmin,xmax]] = (batch_y[i][:,[xmin,xmax]] * (resize[1] / img_width)).astype(np.int)
                            batch_y[i][:,[ymin,ymax]] = (batch_y[i][:,[ymin,ymax]] * (resize[0] / img_height)).astype(np.int)
                            #img_width, img_height = resize # Updating these at this point is unnecessary, but it's one fewer source of error if this method gets expanded in the future

                        if gray:
                            batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), axis=2)

                    # If any batch items need to be removed because of failed random cropping, remove them now.
                    for j in sorted(batch_items_to_remove, reverse=True):
                        batch_X.pop(j)
                        batch_y.pop(j) # This isn't efficient, but this should hopefully not need to be done often anyway

                    if ssd_box_encoder is None:
                        raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                    #这里作主要修改，只返回类别信息
                    y_true = ssd_box_encoder.encode_y(batch_y,isbox = True) #不带box信息
                    yield (np.array(batch_X), y_true)
                else:
                    for i in range(len(batch_X)):
                        batch_X[i] = batch_X[i] / 255.0
                        img_height, img_width, ch = batch_X[i].shape
                        batch_y[i] = np.array(batch_y[i])
                        batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))

                        batch_y[i][:,[xmin,xmax]] = (batch_y[i][:,[xmin,xmax]] * (resize[1] / img_width)).astype(np.int)
                        batch_y[i][:,[ymin,ymax]] = (batch_y[i][:,[ymin,ymax]] * (resize[0] / img_height)).astype(np.int)
                    y_true = ssd_box_encoder.encode_y(batch_y)   #不带box信息
                    # y_true = ssd_box_encoder.encode_y(batch_y)  #对坐标进行编码
                    yield (np.array(batch_X), y_true)
            #非测试过程,只对图像作resize变换
            else:
                Row_batch_X = np.copy(batch_X)
                Row_batch_y = np.copy(batch_y)
                for i in range(len(batch_X)):
                    batch_X[i] = batch_X[i] / 255.0
                    img_height, img_width, ch = batch_X[i].shape
                    batch_y[i] = np.array(batch_y[i])
                    batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))

                    batch_y[i][:,[xmin,xmax]] = (batch_y[i][:,[xmin,xmax]] * (resize[1] / img_width)).astype(np.int)  #神经病
                    batch_y[i][:,[ymin,ymax]] = (batch_y[i][:,[ymin,ymax]] * (resize[0] / img_height)).astype(np.int)
                y_true = ssd_box_encoder.encode_y(batch_y,isbox = True)   #带box信息
                # y_true = ssd_box_encoder.encode_y(batch_y)  #对坐标进行编码
                yield (np.array(batch_X), y_true, Row_batch_X,Row_batch_y, this_filenames)


    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames and the list of labels.
        '''
        return self.filenames, self.labels

    def get_n_samples(self):
        '''
        Returns:
            The number of image files in the initialized dataset.
        '''
        return len(self.filenames)
if __name__=='__main__':
    from ssd_box_encode_decode_utils import *

    aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
                            [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                            [0.5, 1.0, 2.0]]
    scales=[ 0.1, 0.17, 0.26, 0.35]     #以224为基准
    normalize_coords = True
    ImageSize = (224,224)
    limit_boxes = True
    two_boxes_for_ar1 = True
    coords = 'centroids'
    variances=[0.1, 0.1, 0.2, 0.2]
    classes = ["background","tumor"]
    ssd_box_encoder = SSDBoxEncoder(img_height=ImageSize[0],
                                    img_width=ImageSize[0],
                                    n_classes=2,
                                    predictor_sizes = np.array([[56, 56],
                                                               [28, 28],
                                                               [14, 14]]),
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
                                                          image_set_paths=[r"E:\data_base\breast\test.txt"],
                                                          classes = classes,
                                                          ret = True)
    train_generator = generator0.generate(
                                         batch_size=30,
                                         train=True,Transform=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         #brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(ImageSize[0], ImageSize[1], 1, 3),
                                         full_crop_and_resize=(ImageSize[0], ImageSize[1], 1, 3, 0.5),
                                         random_crop=False,
                                         crop=False,
                                         resize=(224,224),
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4)
    for sa in train_generator:
        #batch_X, y_true, Row_batch_X,Row_batch_y, this_filenames = sa
        batch_X, y_true= sa
        y_pred_decoded = decode_y(y_true,
                          confidence_thresh=0.5,		#用于预选框筛选
                          iou_threshold=0.80,          #用于非极大值抑制
                          top_k=1,                     #我们的样本最多只有3个目标
                          input_coords='centroids',
                          normalize_coords=normalize_coords, #从归一化坐标恢复
                          img_height=ImageSize[0],
                          img_width=ImageSize[1])
        y_temp = y_pred_decoded[0]       #nbox*(2+4),            (background,tumor,coords)
        sort_index = np.argsort(y_temp[:,1])
        y_temp = y_temp[sort_index,:]      #按照肿瘤的置信度排序
        indexmax  = 0

        #显示置信度最大的一个框
        plt.figure(1)
        plt.imshow(batch_X[indexmax])
        current_axis = plt.gca()

        for kk in range(y_temp.shape[0]):
            y_temp0 = y_temp[kk]
            label = '{}: {:.2f}'.format(classes[int(y_temp0[0])], y_temp0[1])
            current_axis.add_patch(plt.Rectangle((y_temp0[2], y_temp0[4]), y_temp0[3]-y_temp0[2], y_temp0[5]-y_temp0[4], color='blue', fill=False, linewidth=2))
            current_axis.text(y_temp0[2], y_temp0[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

        print(sa[1].shape)
    print('over!')