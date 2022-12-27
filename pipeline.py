### IMAGE PREPROCESSING AND MODEL DEVELOPMENT/EVALUATION PIPELINE

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
from skimage.measure import block_reduce
from image_preprocessing import standardize_image_dataset,binarize_dataset,crop_dataset,process_dataset_blur,do_pooling_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.base import clone

# # Image Preprocessing, Model Evaluation/Development Pipeline
# The following pipeline will allow the user to pass in training data, declare preprocessing steps, and pass in a model alongside potential hyperparameters. The pipeline will then apply preprocessing steps to the training data, and then identify the best model/hyperparameter combination given the training data. Best in this case is the ability to maximize F1 score, balancing precision and recall ability for detecting class 1 (cancerous images)

# In[3]:


class model_pipeline:
    
    def __init__(self):
        self = self
    
    def evaluate(self,X,y,preprocessing,model,param_grid,optimizing_metric,n_splits,
                return_transformed_features = True, return_grid = True, return_score = True, return_best_estimator = True,
                return_best_params = True,return_oos_pred = True, return_oos_prob = True, return_threshold_analysis=True):
        
        '''
        The evaluate method allows the user to pass in labeled data (X,y) into the modeling pipeline 
        alongside user defined preprocessing steps, a model, and acceptable hyperparameters for the model.
        
        First, the method will apply preprocessing steps in the order they are defined. If no preprocessing steps
        are necessary, then store an empty list: preprocesssing = []. By applying steps in the order they are defined
        rather than presetting the order, this allows greater control and customization over the general image
        cleaning process while also accounting for the ability of mixing orders in potentially improving resulting
        model performance.
        
        To correctly execute an internal preprocessing step, pass in a tuple where index 0 contains one of 'binarize', 'crop',
        'blur', and 'pool' while index 1 contains a list in order that defines values for all acceptable function
        parameters for that specific preprocessing step. For example to execute:
            - Binarization/Shade Removal, pass in ('binarize',[automate_threshold,t]) into the preprocessing list
            - Cropping/Edge Detection, pass in ('crop',[old_dim,new_dim]) into the preprocessing list
            - Blurring, pass in ('blur',[blur_type,dimension,kernel,sigma_x,sigma_y]) into the preprocessing list
            - Pooling, pass in ('pool',[pool_size,pooling_function]) into the preprocessing list
            
        The user can combine as many or as few of the above preprocessing steps in an order of their choice. As a final
        example, in order to execute binarization, cropping, blurring, and pooling on the feature data IN THAT ORDER, 
        the preprocessing list would have the following format:
            - preprocessing = [('binarize',[automate_threshold,t]),
                                 ('crop',[old_dim,new_dim]),
                                 ('blur',[blur_type,dimension,kernel,sigma_x,sigma_y]),
                                 ('pool',[pool_size,pooling_function])]
        which with explicity defined values could be productionized with:
            - preprocessing = [('binarize',[True,0.3]),
                                 ('crop',[(256,256),(256,256)]),
                                 ('blur',['g',(256,256),(5,5),0,0]),
                                 ('pool',[(2,2),np.max])]
        
        Following the successful completion of preprocessing steps, create a grid search cv object that defines the model
        for evaluation alongside a parameter grid and optimizing metric. This object will be fit on the preprocessed data in 
        order to find the model/hyperparameter combination that achieves the best performance on out of sample data given
        our preprocessing, via the internal cross validation process, where best is defined as maximizing our metric 
        of choice. F1 score is our metric of choice in this case, so unless otherwise defined, 'f1' should be passed in for the
        optimizing_metric value.
        
        The user will define an n_splits value which dictates the number of folds the training set will be split into,
        and the number of iterations for our cross validation procedure. During each interation, the model will be fit
        on n_splits - 1 folds and evaluated on the hold out fold. Overall, this will allow us to evaluate how well our
        model and associated hyperparameter generalizes to out of sample data without incurring sampling bias as a 
        result of a poor train test split.
        
        Following fitting feature data on the grid search object, the grid search will autonomously identify a hyperparameter
        combination that allows the model to generalize best to out of sample data according to our optimizing metric. At this 
        point, we can define what we would like returned.
        
        Readily available return options include:
            
            - returning transformed features
            - returning the entire grid search object
            - returning the best score of the grid search object
            - returning the best estimator of the grid search object
            - returning best parameters that optimize our metric given preprocessing steps and model
        
        One issue with grid search CV is it does not return out of sample predictions for our best model.
        In order to circumvent this, we have a boolean command that lets the pipeline know to execute another 
        k fold cross validation procedure in order to store out of sample predictions for our best performing model. Note:
        Due to discrepencies between random states and sampling, this might yield slightly different results to the 
        best score for our best estimator. We will include options that allow for returning out of sample predictions (0/1) 
        and out of sample predicted probabilities (P(Y = 1 | X))
        
        Finally, assuming we returned out of sample probabilities, we can also execute a threshold analysis to identify whether
        a specific probability threshold allows us to achieve an improved out of sample metric score
        '''
        
        features = X.copy() #copy feature matrix to avoid broadcasting
        
        #apply preprocessing steps as defined
        if len(preprocessing) != 0:
            for a,b in preprocessing:
                if a == 'binarize':
                    features = binarize_dataset(features,b[0],b[1])
                elif a == 'crop':
                    features = crop_dataset(features,b[0],b[1])
                elif a == 'blur':
                    features = process_dataset_blur(features.astype('float32'),b[0],b[1],b[2],b[3],b[4])
                elif a == 'pool':
                    features = do_pooling_dataset(features,b[0],b[1])
        
        #instantiate grid search object
        grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = optimizing_metric,
                                  cv = n_splits)
        #Fit grid search on preprocessed features and labels to ID optimal model/hyperparameter combination that generalizes
        #best to out of sample data
        grid_search.fit(features,y)
        
        #With grid search fit, include initial return values in return dictionary
        return_dict = {}
        
        if return_transformed_features: #return preprocessed features to avoid redundant preprocessing if those steps have
            #been IDd as optimal
            return_dict['features'] = features
        if return_grid: #return entire grid search object
            return_dict['grid_search'] = grid_search
        if return_best_estimator: #return best estimator that has already been fit on all the feature data
            return_dict['best_estimator'] = grid_search.best_estimator_
        if return_best_params: #return optimal parameters that enable model to best generalize
            return_dict['best_params'] = grid_search.best_params_
        if return_score: #return best score
            return_dict['best_score'] = grid_search.best_score_
        
        if return_oos_pred or return_oos_prob: #return out of sample prediction/predicted probabilities for our best model
            preds = y.copy()
            probs = y.copy()
            kfold = KFold(n_splits=n_splits) #cv object
            model_copy = clone(grid_search.best_estimator_) #create copy of best estimator
            for train,test in kfold.split(features):
                xtrain,xtest,ytrain,ytest = features.iloc[train],features.iloc[test],y.iloc[train],y.iloc[test]
                model_copy.fit(xtrain,ytrain)
                preds.iloc[test] = list(model_copy.predict(xtest))
                probs.iloc[test] = list(model_copy.predict_proba(xtest)[:,1])
            
            #store out of sample binary predictions and out of sample probabilities
            if return_oos_pred:
                return_dict['oos_preds'] = preds
            if return_oos_prob:
                return_dict['oos_probs'] = probs
                
                #conduct threshold analysis to identify threshold that optimizes out of sample metric score
                if return_threshold_analysis:
                    best_thresh = 0.5
                    best_score = f1_score(y,preds)
                    best_preds = preds
                    
                    for num in np.linspace(0.01,0.99,99):
                        threshold = num #create threshold
                        mod_preds = np.array([1 if x >= threshold else 0 for x in list(probs)]) #map prob to prediction according to threshold
                        score = f1_score(y,mod_preds) #evaluate out of sample f1 score given threshold and predicted probabilities
                        
                        if score > best_score: #if score using threshold is better than current best, update threshold, score,
                            #and out of sample prediction values using that threshold
                            best_thresh = threshold
                            best_score = score
                            best_preds = mod_preds
                    
                    return_dict['threshold_analysis'] = {'best_thresh':best_thresh,'best_score':best_score,'best_preds':best_preds}
            
        return return_dict