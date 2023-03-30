from PIL import Image
import cv2
import os
import pandas as pd
import imageio as iio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


path = 'D:/Repsonsible_ai/CUB_200_2011/'


def load_data(path, len):
    """
    Args
    - path: str. path to the dataset.
    - len: int. number of images to load.
    Load data set and crop. Put images into train and test folders.
    .datasets/CUB200_cropped/
    + test_cropped
    or train_cropped
    """
    images = pd.read_csv((path + 'images.txt'), sep=' ', names=['img_id', 'filepath'])
    image_class_labels = pd.read_csv(os.path.join(path+'/image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
    train_test_split = pd.read_csv(os.path.join(path+'/train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
    data = images.merge(image_class_labels, on='img_id')
    data = data.merge(train_test_split, on='img_id')
    bounding_box = pd.read_csv(os.path.join(path+'/bounding_boxes.txt'), sep=' ', names=['x', 'y', 'w', 'h'])
    datasets_root_dir = './datasets/cub200_cropped/'
    target_dir = datasets_root_dir + 'train_cropped_augmented/'
    
    for i in range(len):
        img = Image.open(os.path.join(path +'/images/' + data['filepath'][i]))
        arr = np.asarray(img)
        im = center_crop(arr, output_size = [bounding_box.iloc[i][2], bounding_box.iloc[i][3]], center = [bounding_box.iloc[i][0], bounding_box.iloc[i][1]])
        pic = Image.fromarray(im)
        if data['is_training_img'][i] == 1:
            target_dir = datasets_root_dir + 'train_cropped/' + data.loc[i]['filepath'].split('/')[0] + '/'
            makedir(target_dir)
            pic.save(target_dir + data.loc[i]['filepath'].split('/')[1], 'JPEG')
        else:
            target_dir = datasets_root_dir + 'test_cropped/' + data.loc[i]['filepath'].split('/')[0] + '/'
            makedir(target_dir)
            pic.save(target_dir + data.loc[i]['filepath'].split('/')[1], 'JPEG')
    
    return None

def center_crop (img, output_size, center=None, scale=None):
    """
    Args
    - img: np.ndarray. img.shape=[height,width,3].
    - output_size: tuple or list. output_size=[height, width].
    - center: tuple or list. center=[x,y].
                If cenetr is None, use input image center.
    - scale: float.
                If scale is None, do not scale up the boundary.
    """
    hi,wi = img.shape[:2]
    ho,wo = output_size
    if center:
        x,y = center
        if scale:
            ho *= scale
            wo *= scale
            if ho > hi or wo > wi:
                ho,wo = output_size
        if ho > hi or wo > wi:
            ho = min([hi,wi])
            wo = min([hi,wi])
        bound_left  = int(x - wo/2)
        bound_right = int(x + wo/2)
        bound_top   = int(y - ho/2)
        bound_bottom= int(y + ho/2)

        if bound_left < 0:
            offset_w = 0
        elif bound_right > wi:
            offset_w = int(wi-wo)
        else:
            offset_w = int(x - wo/2)

        if bound_top < 0:
            offset_h = 0
        elif bound_bottom > hi:
            offset_h = int(hi-ho)
        else:
            offset_h = int(y - ho/2)
    else:
        if scale:
            print ("Scaling deny when center variable is None.")
        try:
            if hi < ho and wi < wo:
                raise ValueError("image is too small. use orginal image.")
        except:
            ho = min([hi,wi])
            wo = ho
        offset_h = int((hi - ho) / 2)
        offset_w = int((wi - wo) / 2)
    return img[offset_h : offset_h + int(ho),
                offset_w : offset_w + int(wo)]

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)  
    
    
if __name__ == '__main__':
    load_data(path, 100)