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
import gc
import skimage.filters
import cv2
import watermark
import joblib
import math
from skimage.measure import block_reduce
from image_preprocessing import standardize_image_dataset,resize_dataset,binarize_dataset,crop_dataset,process_dataset_blur,do_pooling_dataset
from pipeline import model_pipeline

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.base import clone



def automate_optimal_model_dev(X,y,model,param_grid,preprocessing_eval_order = ['bin/crop','blur','pool'],resize=True):
    
    #Store global values to capture optimal model/parameters, key performance metrics, and optimal preprocessing steps
    best_feats = X.copy()
    best_model = None
    best_params = None
    best_probs = None
    best_preds = None
    best_thresh = None
    best_score = 0
    best_preprocess = '(Initial Standardization/Resizing to ' + str((int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))) + ')'
    # Initial Evaluation - Identify Optimal Size of Images, measured by performance of optimal model yielded by training on 
    #images of various
    
    if resize == True:

        for img_size in [(2**num,2**num) for num in range(4,int(math.log2(np.sqrt(X.shape[1]))) + 1)]:

            resize_results = model_pipeline().evaluate(resize_dataset(X,(int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1]))),img_size),
                                                       y,preprocessing=[],model=model,param_grid=param_grid,
                                                       optimizing_metric='f1',n_splits=5,return_transformed_features=True,
                                                       return_grid=True,return_score=True,return_best_estimator=True,
                                                       return_best_params=True,return_oos_pred=True,return_oos_prob=True,
                                                       return_threshold_analysis=True)
            score = resize_results['threshold_analysis']['best_score'] #get f1 score of best performing model trained on images
            #of specified size

            if score - best_score > 0.001:

            #Extract key results and the model for best performing model if model is at minimum > 0.01 in F1 score performance
            # then previously identified optimal model
                best_feats = resize_results['features']
                best_model = resize_results['best_estimator']
                best_params = resize_results['best_params']
                best_probs = resize_results['oos_probs']
                best_preds = resize_results['threshold_analysis']['best_preds']
                best_thresh = resize_results['threshold_analysis']['best_thresh']
                best_score = score
                best_preprocess = '(Initial Standardization/Resizing to ' + str(img_size) + ')'
                print('Better Model Identified by Resizing Images to ' + str(img_size) + ': ' + str(score))
            else: 
                #break early if increasing image size does not yield a significantly better performing optimal model
                break
    
    #detect size of images for binarization/cropping and blur preprocessing steps
    image_size = int(np.sqrt(best_feats.shape[1]))
    
    binarization_crop_settings = [[('binarize',[True,0.3]),('crop',[(image_size,image_size),(image_size,image_size)])],
                                         [('binarize',[False,0.05]),('crop',[(image_size,image_size),(image_size,image_size)])],
                                         [('binarize',[False,0.1]),('crop',[(image_size,image_size),(image_size,image_size)])],
                                         [('binarize',[False,0.15]),('crop',[(image_size,image_size),(image_size,image_size)])],
                                         [('binarize',[False,0.2]),('crop',[(image_size,image_size),(image_size,image_size)])],
                                         [('binarize',[False,0.3]),('crop',[(image_size,image_size),(image_size,image_size)])]]
    
    blur_settings = [[('blur',['g',(image_size,image_size),(3,3),0,0])],
                     [('blur',['g',(image_size,image_size),(3,3),1,0])],
                     [('blur',['g',(image_size,image_size),(3,3),0,1])],
                     [('blur',['g',(image_size,image_size),(3,3),1,1])],
                     [('blur',['g',(image_size,image_size),(3,3),2,2])],
                     [('blur',['g',(image_size,image_size),(5,5),0,0])],
                     [('blur',['g',(image_size,image_size),(5,5),1,0])],
                     [('blur',['g',(image_size,image_size),(5,5),0,1])],
                     [('blur',['g',(image_size,image_size),(5,5),1,1])],
                     [('blur',['g',(image_size,image_size),(5,5),2,2])],
                     [('blur',['b',(image_size,image_size),(3,3),0,0])],
                     [('blur',['b',(image_size,image_size),(3,3),1,0])],
                     [('blur',['b',(image_size,image_size),(3,3),0,1])],
                     [('blur',['b',(image_size,image_size),(3,3),1,1])],
                     [('blur',['b',(image_size,image_size),(3,3),2,2])],
                     [('blur',['b',(image_size,image_size),(5,5),0,0])],
                     [('blur',['b',(image_size,image_size),(5,5),1,0])],
                     [('blur',['b',(image_size,image_size),(5,5),0,1])],
                     [('blur',['b',(image_size,image_size),(5,5),1,1])],
                     [('blur',['b',(image_size,image_size),(5,5),2,2])]]
    
    
    #detect possible pooling settings dependent on image_size, controls possible pool sizes
    pool_ranges = int(math.log2(image_size))
    pool_settings = []
    for num in range(1,pool_ranges):
        pool_settings.append([('pool',[(2**num,2**num),np.max])])
        pool_settings.append([('pool',[(2**num,2**num),np.mean])])
    
    for step in preprocessing_eval_order:
        #Identify optimal model considering different image preprocessing settings to also identify
        #optimal preprocessing settings
        
        #set settings we will evaluate depending on the user defined preprocessing evaluation order
        if step == 'bin/crop':
            settings = binarization_crop_settings
        elif step == 'blur':
            settings = blur_settings
        elif step == 'pool':
            settings = pool_settings
        
        best_setting = ''
        best_setting_feats = None

        #For each preprocessing setting, identify an optimal performing model trained on
        #transformed features according to specified preprocessing. Compare each model to currently identified
        #optimal model and replace if better model is found
        for setting in settings:
            if step == 'bin/crop' and int(np.sqrt(best_feats.shape[1])) != image_size: #if pooling was evaluated first and
                #yielded a model better than base case, resulting data would have been resized so dimension settings for
                #binarization, cropping will need to be adjusted
                new_image_size = int(np.sqrt(best_feats.shape[1]))
                setting[1][1][0] = (new_image_size,new_image_size)
                setting[1][1][1] = (new_image_size,new_image_size)
            elif step == 'blur' and int(np.sqrt(best_feats.shape[1])) != image_size: #same case as above but for blurring
                new_image_size = int(np.sqrt(best_feats.shape[1]))
                setting[0][1][1] = (new_image_size,new_image_size)
            setting_case = model_pipeline().evaluate(best_feats,y,preprocessing=setting,model=model,param_grid=param_grid,
                                                      optimizing_metric='f1',n_splits=5,return_transformed_features=True,
                                                      return_grid=True,return_score=True,return_best_estimator=True,
                                                      return_best_params=True,return_oos_pred=True,return_oos_prob=True,
                                                      return_threshold_analysis=True)
            score = setting_case['threshold_analysis']['best_score']#get F1 score of optimal model trained using preprocessed features
            if score > best_score: #if score is better than current best score, update key results and model for optimal performing model
                best_model = setting_case['best_estimator']
                best_params = setting_case['best_params']
                best_probs = setting_case['oos_probs']
                best_preds = setting_case['threshold_analysis']['best_preds']
                best_thresh = setting_case['threshold_analysis']['best_thresh']
                best_score = score
                best_setting_feats = setting_case['features']
                if step == 'bin/crop':
                    best_setting = '(Binarization, Automate Threshold = ' + str(setting[0][1][0]) + ', Threshold = ' + str(setting[0][1][1]) + ') (Crop, ' + str(setting[1][1][0]) + ', ' + str(setting[1][1][0]) + ')'
                    print('Better Model Identified W/ Binarization/Cropping, Score = ' + str(score))
                elif step == 'blur':
                    best_setting = '(Blurring, Type = ' + str(setting[0][1][0]) + ', Dimension = ' + str(setting[0][1][1]) + ', Kernel = ' + str(setting[0][1][2]) + ', sigma_x = ' + str(setting[0][1][3]) + ', sigma_y = ' + str(setting[0][1][4]) + ')'
                    print('Better Model Identified W/ Blurring, Score = ' + str(score))
                elif step == 'pool':
                    best_setting = '(Pool, pool_size = ' + str(setting[0][1][0]) + ', pooling_function = ' + str(setting[0][1][1]) + ')'
                    print('Better Model Identified W/ Pooling, Score = ' + str(score))
                    

        #Update features and preprocessing string if incorporating specific preprocessing as part of image preprocessing pipeline yielded 
        #a better performing model. This ensures these steps do not need to be repeated when evaluating additional 
        #preprocessing steps
        if best_setting != '':
            best_feats = best_setting_feats
            best_preprocess = best_preprocess + best_setting
    
    
    #store and return optimal model, threshold, out of sample predictions, features the model was trained on, 
    #and optimal preprocessing steps identified via a greedy sequential decision process
    return_dict = {}
    return_dict['features'] = best_feats
    return_dict['best_model'] = best_model
    return_dict['best_params'] = best_params
    return_dict['oos_probs'] = best_probs
    return_dict['oos_preds'] = best_preds
    return_dict['best_thresh'] = best_thresh
    return_dict['best_score'] = best_score
    return_dict['best_preprocess'] = best_preprocess
    
    return return_dict