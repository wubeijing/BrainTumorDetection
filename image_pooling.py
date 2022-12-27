import pickle
import pandas as pd
import numpy as np
import cv2
import watermark
from matplotlib import pyplot as plt
import time
import joblib
from skimage.measure import block_reduce

# scan_path = "/Users/neilbhatia/GitHub/w207_final_project/Amit/Labeled Data/labeled_data.pkl"
# brain_scans = pickle.load(open(scan_path,'rb'))
# original_img = brain_scans.iloc[192,:-1].to_numpy().reshape(150, 150)




def do_pooling(image, pool_size = (2,2), pooling_function = np.max):
    """This function takes an image pixel matrix and, given a pool size
    and pooling function, returns a pooled version of that image"""
    
    if pooling_function in [np.mean,np.max,np.min]:
        return block_reduce(image, block_size=pool_size, func = pooling_function) 
    else:
        raise Exception("Please specify a pooling function, either np.mean, np.max, or np.min.")


# pooled_image =  do_pooling(original_img,pool_size=(2,2),pooling_function=np.max)


def do_pooling_dataset(image_dataset, pool_size = (2,2), pooling_function = np.max):
    pooled_images = image_dataset.copy()
    for num in range(len(pooled_images)):
        image_reshaped = pooled_images.iloc[num].to_numpy().reshape(150,150)                         
        pooled_images.iloc[num] = do_pooling(image_reshaped,pool_size,pooling_function).flatten()
    return pooled_images