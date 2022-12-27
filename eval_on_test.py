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

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,CategoricalNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.base import clone
from sklearn.metrics import confusion_matrix


def make_preds(X,y,preprocessing,model,threshold,return_features = True):
        
        features = X.copy() #copy feature matrix to avoid broadcasting
        
        #apply preprocessing steps as defined
        if len(preprocessing) != 0:
            for a,b in preprocessing:
                if a == 'binarize':
                    features = binarize_dataset(features,b[0],b[1])
                elif a == 'resize':
                    features = resize_dataset(features,b[0],b[1])
                elif a == 'crop':
                    features = crop_dataset(features,b[0],b[1])
                elif a == 'blur':
                    features = process_dataset_blur(features.astype('float32'),b[0],b[1],b[2],b[3],b[4])
                elif a == 'pool':
                    features = do_pooling_dataset(features,b[0],b[1])
        
        #make predictions on test data using fit model
        probs = model.predict_proba(features)[:,1]
        preds = np.array([1 if x>=threshold else 0 for x in probs])
        #evaluate performance of model by comparing predicted values to actual values
        f1 = f1_score(y,preds)
        acc = accuracy_score(y,preds)
        confuse_mat = confusion_matrix(y,preds)
        
        #return key metrics of model performance and the model's predictions on test data
        return_dict = {}
        if return_features:
            return_dict['features'] = features
        return_dict['probs'] = probs
        return_dict['preds'] = preds
        return_dict['f1 score'] = f1
        return_dict['accuracy'] = acc
        return_dict['confusion_matrix'] = confuse_mat
        
        return return_dict