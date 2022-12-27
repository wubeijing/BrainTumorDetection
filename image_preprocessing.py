# IMAGE PREPROCESSING FUNCTIONS FOR USE IN MODEL DEVELOPMENT, EVALUATION, AND PRODUCTION
import numpy as np
import pandas as pd
import PIL as pil
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
import tempfile
import pickle
import time
import skimage.filters
import cv2
import watermark
import joblib
from skimage.measure import block_reduce

#---------------------------------------------------------------------------------------------------------------------------#

#STANDARDIZE SINGLE IMAGE
def standardize_image_dimensions(filename,newdim=(360,360),show_difference=False):
    '''
    Ingest image file and output a greyscale version of the image and associated vector representation that is a modified 
    square size while preserving aspect ratios
    '''
    img = pil.Image.open(filename).convert('L') #open image as a grayscale
    img_size = img.size #capture image dimensions
    img_size = (img_size[1],img_size[0]) #store in height by width rather than width by height aka rowsize x colsize
    
    if show_difference == True: #show before standardization
        display(img)
    
    if img_size[0] == img_size[1]: #if image is already square, we can resize while preserving aspect ratios of original image
        img_array = np.array(img) #convert image to array
        
        if show_difference == True:
            plt.imshow(img_array,cmap='gray')
    else:
        gap = abs(img_size[0] - img_size[1]) #calculate difference between dimensions to ID space we need to fill on shorter dimensions
        gapfill_1 = int(gap/2) #space we will fill on one side of shorter dimension
        gapfill_2 = gap-gapfill_1 #space we will fill on other side
        img_array = np.asarray(img) #convert image to array
        
        if img_size[0] < img_size[1]: #If rows are less than columns, fill rowspace with empty ink signal symmetrically s.t. the image is square
            img_array = np.vstack((np.zeros((gapfill_1,img_size[1])),img_array)) #fill top
            img_array = np.vstack((img_array,np.zeros((gapfill_2,img_size[1])))) #fill bottom
        else: #if columns are less than rows, fill column space with empty ink signal symmetrically s.t. the image is square
            img_array = np.hstack((np.zeros((img_size[0],gapfill_1)),img_array)) #fill left
            img_array = np.hstack((img_array,np.zeros((img_size[0],gapfill_2)))) #fill right
        
    #resize new image according to user defined size
    temp = tempfile.TemporaryFile() #create temporary file
    plt.imsave(temp,img_array,cmap='gray') #save array representation of image as image
    new_image = pil.Image.open(temp).convert('L') #open image
    new_image = new_image.resize(newdim) #resize image
    
    if show_difference == True: #show after standardization
        display(new_image)
    
    #return standardized image and vectorized version of standardized image
    return new_image,np.array(new_image)

#-------------------------------------------------------------------------------------------------------------------------------#
#STANDARDIZE ENTIRE IMAGE DATASET
def standardize_image_dataset(filename_list,newdim=(360,360)):
    '''
    Function that takes in list of image file names and a desired square dimensional output for modified images, before returning
    standardized versions of each image in greyscale format at the specified new dimensions. Modified images are stored in local
    directory in addition to labeled dataset for machine learning model development
    '''
    labeled_dataframe = pd.DataFrame(np.zeros((len(filename_list),newdim[0]*newdim[1] + 1))) #create dataframe to that stores
    #vectorized versions of images -- initial labeled dataset
    #Create column names
    colNames = ['pixel' + str(x) for x in range(1,newdim[0]*newdim[1] + 1)]
    colNames.append('label')
    labeled_dataframe.columns = colNames
    
    #counters for saving each image as a unique filename in the proper location
    healthy_counter = 0
    cancerous_counter = 0
    
    for num in range(len(filename_list)):
        filename = filename_list[num] #get filename
        img,img_asarray = standardize_image_dimensions(filename,newdim,False) #get standardized/vectorized version of image
        if '/Cancer' in filename: #if cancer is in filename, save in proper folder and add to labeled data with label = 1
            cancerous_counter = cancerous_counter + 1
            img.save('Amit/Standardized Images/Cancerous/Cancer' + str(cancerous_counter) + '.jpg') #Save in cancer folder
            #Store vectorized version of image as 1 dimensional array
            new_data = list(img_asarray.flatten())
            new_data.append(1)
            labeled_dataframe.iloc[num] = new_data
        else:#if cancer is in filename, save in proper folder and add to labeled data with label = 1
            healthy_counter = healthy_counter + 1
            img.save('Amit/Standardized Images/Healthy/Healthy' + str(healthy_counter) + '.jpg') #Save in cancer folder
            #Store vectorized version of image as 1 dimensional array
            new_data = list(img_asarray.flatten())
            new_data.append(0)
            labeled_dataframe.iloc[num] = new_data
    
    #save labeled data as a dataframe using to_pickle (read_pickle to read dataframe in for further analysis and model development)
    labeled_dataframe = labeled_dataframe.astype('uint8')
    labeled_dataframe.to_pickle('Amit/Labeled Data/labeled_data.pkl')
#-------------------------------------------------------------------------------------------------------------------------------#
#RESIZE SQUARE IMAGE
def resize_vector_image(img_vec,old_dim,new_dim):
    #make 1d image vector a 2d image vector
    img = np.array(img_vec).reshape(old_dim)
    #save as image
    temp = tempfile.TemporaryFile()
    plt.imsave(temp,img,cmap='gray')
    #open as greyscale image
    img = PIL.Image.open(temp).convert('L')
    img = img.resize(new_dim)
    #convert image to 2D array representation
    img = np.array(img)
    #return as 1D array representation to be stored as image features
    return img.flatten()

#-------------------------------------------------------------------------------------------------------------------------------#
#RESIZE ENTIRE DATASET STORING VECTORIZED REPRESENTATIONS OF SQUARE IMAGES
def resize_dataset(image_dataset,old_dim,new_dim,return_type = 'uint8'):
    vectorized_images = []
    for num in range(len(image_dataset)): #resize every vectorized image to new dimensions
        vectorized_images.append(resize_vector_image(image_dataset.iloc[num],old_dim,new_dim))
    #return vectorized images in a dataframe format
    return pd.DataFrame(vectorized_images,index = image_dataset.index,columns = ['pixel' + str(x) for x in range(0,new_dim[0]**2)]).astype(return_type)

#-------------------------------------------------------------------------------------------------------------------------------#
#BINARIZE SINGLE IMAGE
def binarize(img_vector, threshold):
    '''
    Function that takes in an image vector and greyscale signal threshold before returning a modified version of the image vector
    that has mapped all values below the threshold to 0

    '''
    return [0 if pixel <= threshold else pixel for pixel in img_vector]

#-------------------------------------------------------------------------------------------------------------------------------#
#BINARIZE ENTIRE DATASET
def binarize_dataset(dataset, automate_threshold = True, t = 0.3):
    '''
    Function that takes in an entire image dataset represnted via numerical vectors for each image, and binarizes the entire dataset
    using either a user defined threshold or an autonomously setected threshold using skimage

    '''
    #binarize_dataset = dataset.copy()
    binarize_dataset = []
    for num in range(len(dataset)):
        img_vector_norm = dataset.iloc[num] / np.array(dataset.iloc[num]).max()
        if automate_threshold == True:
            t = skimage.filters.threshold_otsu(img_vector_norm)
        binarize_dataset.append(binarize(img_vector_norm, t))
            #binarize_dataset.iloc[num] = binarize(img_vector_norm, t)
        #else:
            #binarize_dataset.iloc[num] = binarize(img_vector_norm, t)
    return pd.DataFrame(binarize_dataset,index=dataset.index,columns=dataset.columns)

#-------------------------------------------------------------------------------------------------------------------------------#
#CROP SINGLE IMAGE
def crop_single_image(image_vector,old_dim,new_dim,show_stages = False):

    '''
    Function that takes in a greyscale image vector, current dimension size, and desired square dimension size before
    using edge detection logic and symmetric padding to return a cropped, centered version of the image vector at the desired
    dimensions
    
    '''
    #Store Vector as a 2D vector (square pixel representation of image)
    image_vector_2d = np.array(image_vector.copy()).reshape(old_dim)
    
    if show_stages == True:
        print('Original Image')
        plt.imshow(image_vector_2d,cmap='gray')
        plt.show()
    
    #Store max ink signal across rows and columns
    row_maxes = np.max(image_vector_2d,axis=1)
    col_maxes = np.max(image_vector_2d,axis=0)
    
    #Find tightest possible bounds for image where ink signal begins and ends
    #LEFT/RIGHT BOUNDS, TOP/BOTTOM BOUNDS
    left_most,right_most,top_most,bottom_most = 0,0,0,0
    switch_col = 0 #toggle to switch between capturing left most ink signal and right most ink signal
    switch_row = 0 #toggle to switch between capturing top most ink signal and bottom most ink signal
    for num in range(len(row_maxes)):
        #LEFT/RIGHT BOUNDS
        if switch_col == 0 and col_maxes[num] != 0:
            left_most = num
            switch_col = 1
        elif switch_col == 1 and col_maxes[num] == 0:
            right_most = num - 1
            switch_col = 2
            
        #TOP/BOTTOM BOUNDS
        if switch_row == 0 and row_maxes[num] != 0:
            top_most = num
            switch_row = 1
        elif switch_row == 1 and row_maxes[num] == 0:
            bottom_most = num - 1
            switch_row = 2
        
        #Early break if cropped bounds have been found
        if switch_col == 2 and switch_row == 2:
            break
        
        #If on last iteration bottom or right bound has not been found
        if num == len(row_maxes) - 1 and switch_col == 1:
            right_most = num
        if num == len(row_maxes) - 1 and switch_row == 1:
            bottom_most = num
        
    #Cropped image vector according to bounds identified
    cropped_image_vector_2d = image_vector_2d[top_most:bottom_most + 1,left_most:right_most + 1].copy()
    
    if show_stages == True:
        print('Cropped Image around top/bottom most and left/right most instances of ink signal')
        plt.imshow(cropped_image_vector_2d,cmap='gray')
        plt.show()
    
    #Cast image as a square image if needed to preserve aspect ratios
    #Capture shape of cropped image and difference in size across dimensions
    height,width = cropped_image_vector_2d.shape
    size_diff = abs(height - width) #diff between height and width dimensions
    gapfill_1 = int(size_diff/2) #amount of blank space to add to one side of shorter dimension
    gapfill_2 = size_diff - gapfill_1 #amount of blank space to add to the other side
    
    if height < width: #if height of image is less than width, add symmetrical padding height wise to make image square
        #Add symmetrical blank spacing to top and bottom of image
        cropped_image_vector_2d = np.vstack((np.zeros((gapfill_1,width)),cropped_image_vector_2d)) #Add to top
        cropped_image_vector_2d = np.vstack((cropped_image_vector_2d,np.zeros((gapfill_2,width)))) #Add to bottom
    elif width < height: #if width of image is less than height, add symmetrical padding width wise to make image square
        cropped_image_vector_2d = np.hstack((np.zeros((height,gapfill_1)),cropped_image_vector_2d)) #Add to left
        cropped_image_vector_2d = np.hstack((cropped_image_vector_2d,np.zeros((height,gapfill_2)))) #Add to right
    
    if show_stages == True and height != width:
        print('Centered Version of Cropped Image with symmetric padding added to short axis')
        plt.imshow(cropped_image_vector_2d,cmap='gray')
        plt.show()
        
    #Resize as needed (only if cropped image is not the size of the new dimensions)
    height,width = cropped_image_vector_2d.shape
    if (height,width) != new_dim:
        temp = tempfile.TemporaryFile()
        plt.imsave(temp,cropped_image_vector_2d,cmap='gray') #save array as image
        img = PIL.Image.open(temp).convert('L') #open greyscale image
        img = img.resize(new_dim) #resize to new dimensions
        cropped_image_vector_2d = np.array(img)
    
    if show_stages == True:
        print('Final Image')
        plt.imshow(cropped_image_vector_2d,cmap='gray')
        plt.show()
    
    return cropped_image_vector_2d

#--------------------------------------------------------------------------------------------------------------------------------#
#CROP ENTIRE DATASET
def crop_dataset(image_dataset,old_dim,new_dim):
    cropped_images = []
    for num in range(len(image_dataset)):
        scan = image_dataset.iloc[num].copy()
        cropped_images.append(crop_single_image(scan,old_dim,new_dim,False).flatten())
    
    if list(image_dataset.dtypes.unique()) == [('uint8')] or list(image_dataset.dtypes.unique()) == [('int8')] :
        cropped_images = pd.DataFrame(cropped_images,columns=image_dataset.columns).astype('uint8')
    else:
        cropped_images = pd.DataFrame(cropped_images,columns=image_dataset.columns).astype('float32')
        
    return cropped_images


#-------------------------------------------------------------------------------------------------------------------------------#
#BLUR SINGULAR IMAGES
def process_image_blur(single_image_array, blur_type='g', kernel=(5,5), sigma_x=0, sigma_y=0):
    """Different Blurring Implementation for a single image
    
    Args:
    
        blur_type: Blur Type, valid values are:
            'g': Apply a Gaussian Filter
            'b': Apply a normalized Box Filter
        
        single_image_array: 2-D numpy.ndarray of image vector representing a gray-scale image.
    
        kernel: Filter Kernel tuple, valid tuple element values are positive, and odd. Default value is (5,5).
                ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be
                zero's and then they are computed from sigma.
    
        sigma_x: Gaussian kernel standard deviation in X direction. 
    
        sigma_y: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal
                to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively.
    
    Returns:
        blurred_image_array: 2-D numpy.ndarray image array blurred with appropriate LPF.
    
    Reference: 
        https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        
    """
    
    if blur_type == 'g':
        blurred_img_array = cv2.GaussianBlur(src=single_image_array, ksize=kernel, sigmaX=sigma_x, sigmaY=sigma_y)
    elif blur_type == 'b':
        blurred_img_array = cv2.blur(src=single_image_array, ksize=kernel)
    return blurred_img_array

#-------------------------------------------------------------------------------------------------------------------------------#
#BLUR ENTIRE DATASET
def process_dataset_blur(image_dataset, blur_type='g', dimension=(150, 150), kernel=(5,5), sigma_x=0, sigma_y=0):
    """Different Blurring Implementation for a whole image dataset
    
    Args:
    
        blur_type: Blur Type, valid values are:
            'g': Apply a Gaussian Filter
            'b': Apply a normalized Box Filter
        
        image_dataset: dataframe of image vectors (each image is a flattened row in the dataframe)
        
        dimension: Dimension tuple of each image. This is assumed that prior to calling blurring, image dataset is 
                standardized to a uniform dimension. Default dimension is (150,150)
            
        kernel: Filter Kernel tuple, valid tuple element values are positive, and odd. Default value is (5,5).
                ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be
                zero's and then they are computed from sigma.
    
        sigma_x: Gaussian kernel standard deviation in X direction. 
    
        sigma_y: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal
                to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively.
    
    Returns:
        blurred_image_dataframe: a Dataframe of blurred images, where each row is a flattened 2-D numpy.ndarray 
            containing a single image.
    
    Reference: 
        https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        
    """
    smoothed_images = image_dataset.copy()
    for num in range(len(smoothed_images)):
        back_to_2d_image = smoothed_images.iloc[num].to_numpy().reshape(dimension)                         
        smoothed_images.iloc[num] = process_image_blur(back_to_2d_image, blur_type, kernel=kernel, sigma_x=sigma_x, sigma_y=sigma_y).flatten()
    return smoothed_images

#---------------------------------------------------------------------------------------------------------------------------------#
#POOL SINGULAR IMAGE
def do_pooling(image, pool_size = (2,2), pooling_function = np.max):
    """This function takes an image pixel matrix and, given a pool size
    and pooling function, returns a pooled version of that image"""
    
    if pooling_function in [np.mean,np.max,np.min]:
        return block_reduce(image, block_size=pool_size, func = pooling_function) 
    else:
        raise Exception("Please specify a pooling function, either np.mean, np.max, or np.min.")

#---------------------------------------------------------------------------------------------------------------------------------#
#POOL ENTIRE DATASET

def do_pooling_dataset(image_dataset, pool_size = (2,2), pooling_function = np.max):
    pooled_images = []
    img_size = int(np.sqrt(image_dataset.shape[1]))
    for num in range(len(image_dataset)):
        image_reshaped = image_dataset.iloc[num].to_numpy().reshape(img_size,img_size)                         
        pooled_images.append(do_pooling(image_reshaped,pool_size,pooling_function).flatten())
    return pd.DataFrame(pooled_images,index=image_dataset.index)