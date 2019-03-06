#---------------------------------------------------------------------------
# Utility functions used in the ABCD Neurocognitive Prediction Challenge 
# (ABCD-NP-Challenge 2019) 
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# All imports here
#---------------------------------------------------------------------------

from keras.layers import Flatten, Dense, Dropout, ZeroPadding3D, Conv3D, Activation, MaxPooling3D,\
    GlobalAveragePooling3D, Average, Input
from keras.models import Model
from keras.regularizers import l2, l1
import nibabel as nib
import random
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization


#---------------------------------------------------------------------------
# All class definitions here
#---------------------------------------------------------------------------

                
#---------------------------------------------------------------------------
# All function definitions here
#---------------------------------------------------------------------------

#=======================================================================
def get_model(Shape, weight_decay=0.00005, kernel_initializer='glorot_uniform', reg='l2'):
    # Returns a Keras regression CNN model. Based on:
    # Cole, James H., et al. "Predicting brain age with deep learning f
    # rom raw imaging data results in a reliable and heritable 
    # biomarker." NeuroImage 163 (2017): 115-124.
    #___________________________________________________________________]
    
    if reg == 'l1':
        reg_func = l1
    else:
        reg_func = l2
    
    input_layer = Input(shape=Shape)
    x = Conv3D(8, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv1',
               kernel_regularizer=reg_func(weight_decay))(input_layer)
    x = Activation('relu')(x)
    x = Conv3D(8, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv2',
               kernel_regularizer=reg_func(weight_decay))(x)
    x = BatchNormalization(momentum=0.99)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    x = Conv3D(16, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv3',
               kernel_regularizer=reg_func(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv4',
               kernel_regularizer=reg_func(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    x = Conv3D(32, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv5',
               kernel_regularizer=reg_func(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='valid', kernel_initializer=kernel_initializer, name='conv6',
               kernel_regularizer=reg_func(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    x = Flatten()(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.3)(x)
#     x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.3)(x)
    predictions = Dense(1, name='fcdense', kernel_initializer='he_normal')(x)

    return Model(input=input_layer, output=predictions)
#=======================================================================

#=======================================================================
def data_generator(X, y, Shape, BatchSize=8):
    # Returns a Keras generator.
    # X     - [N x C] Either a list of NIfTIs (single-channel), or a list 
    #                 of lists of NIfTIs (multi-channel)
    # y     - [N x 1] A list of regression targets
    # Shape - Size of a training sample
    #___________________________________________________________________

    while True:
        batch_idx = 0
        # At each epoch shuffle the data differently, but make sure x and y match
        shuffled_index = list(range(len(X)))
        random.shuffle(shuffled_index)

        for i in shuffled_index:
            x1 = np.zeros((BatchSize,) + Shape, dtype=np.float32)
            y1 = np.zeros((BatchSize, 1), dtype=np.float32)
               
            img = get_img(X[i],Shape)
            
            x1[batch_idx % BatchSize] = img
            y1[batch_idx % BatchSize] = y[i]
            batch_idx += 1
            
            if (batch_idx % BatchSize) == 0:
                yield (x1, y1)
#=======================================================================       

#=======================================================================
def get_abcd_training_data_img(Dir,Scores):
    # Returns training data for the ABCD challenge, where Dir is a 
    # directory of NIfTIs, and Scores is a dictionary that contains the
    # residual IQ scores.
    #___________________________________________________________________
        
    X = []
    y = []
    for root, directories, filenames in os.walk(Dir):
        for filename in filenames:                         
            ix   = filename.find('NDAR')
            Name = filename[ix:ix + 16]

            # Targets
            y.append(Scores[Name])
            
            # Images
            X.append(nib.load(os.path.join(root,filename)))
            
    Shape = X[0].header.get_data_shape()
    Shape = (Shape[0],Shape[1],Shape[2],1)
            
    return X,y,Shape
#=======================================================================

#=======================================================================
def get_abcd_training_data_seg(Dir,Scores):
    # Returns training data for the ABCD challenge, where Dir is a 
    # directory of directories, where each inner directory contains a 
    # set of NIfTIs (e.g., segmentations), and Scores is a dictionary 
    # that contains the residual IQ scores.
    #___________________________________________________________________
           
    X   = []
    y   = []
    for root0, directories0, filenames0 in os.walk(Dir):
    
        for directory in directories0:
        
            X1 = []
            for root1, directories1, filenames1 in os.walk(os.path.join(root0,directory)):
            
                for filename in filenames1:                         
                    # Images
                    X1.append(nib.load(os.path.join(root0,directory,filename)))
                    
                ix   = filename.find('NDAR')
                Name = filename[ix:ix + 16]

                # Images
                X.append(X1)
                
                # Targets
                y.append(Scores[Name])                  
    
    Shape = X[0][0].header.get_data_shape()
    Shape = (Shape[0],Shape[1],Shape[2],len(X[0]))
            
    return X,y,Shape
#=======================================================================

#=======================================================================    
def get_abcd_scores(PthScores):
    # Returns a dictionary that maps ABCD subject names to residual IQ
    # scores as ScoresDict[Name] = Score. The scores are read from the
    # .csv-file in PthScores.
    #___________________________________________________________________
    
    with open(PthScores, 'rt') as f:
        reader = csv.reader(f)
        Scores = list(reader)

    Scores = Scores[1:]

    ScoresDict = {}
    for i in range(0,len(Scores)):
        ScoresDict[Scores[i][0]] = float(Scores[i][1])     
        
    return ScoresDict
#=======================================================================    

#=======================================================================    
def plot_histogram(x):
    # Just plots a histogram of the data in vector x
    #___________________________________________________________________
    
    mu = np.mean(x)
    sd = np.std(x)
    
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('y')
    plt.title('ABCD Targets')
#=======================================================================

#=======================================================================
def get_nii_data(nii):
    # More memory efficient way of getting NIfTI data, see:
    # https://nipy.org/nibabel/images_and_memory.html
    #___________________________________________________________________
    
    return np.asarray(nii.dataobj)
#=======================================================================

#=======================================================================
def get_img(X,Shape):
    # Get a, possibly, multi-channel image
    #___________________________________________________________________
    
    if Shape[-1] > 1:
        # Multi-channel data
        img = np.zeros(Shape, dtype=np.float32)
        for i1 in range(0,len(X)):
            img[:,:,:,i1] = get_nii_data(X[i1])
    else:
        # Single-channel data
        img = get_nii_data(X)
        img = np.expand_dims(img, axis=-1)

    img[np.isnan(img)] = 0
    img[img < 0]       = 0

#     img = img / img.max()
    
    return img
#=======================================================================

#=======================================================================        
def imshow(X,Shape,ix=np.arange(1)):
    # Show a bunch of images
    #___________________________________________________________________
    
    N = len(ix)
    
    plt.figure(1)
    if N == 1:
        # Single plot
        plt.imshow(get_slice(X[ix[0]],Shape))
        plt.axis('off') 
    else:
        # Subplot  
        nc = np.floor(np.sqrt(N));
        nr = np.ceil(N/nc);  
        
        for i in range(0,N):
            plt.subplot(nr,nc,i + 1)            
            plt.imshow(get_slice(X[ix[i]],Shape))
            plt.axis('off') 
    
    plt.show()    
#=======================================================================    

#=======================================================================
def get_slice(X,Shape):
    # Get slice(s) in the z-direction
    #___________________________________________________________________    
    
    N  = len(X)     
    x  = get_img(X,Shape)
    x  = np.squeeze(x) 
    dm = x.shape            

    is3d = dm[2] > 1
    if is3d:
        z = int(np.ceil(dm[2]/2))
    else:
        z = 1
        
    x = x[:,:,z]                 
    x = np.squeeze(x) 
    
    if N == 1:
        x = np.rot90(x)
    else:                     
        x = np.transpose(x,(1, 0, 2))
        x = np.rot90(x)
        x = np.rot90(x)
        
        x = np.reshape(x, (Shape[1],N*Shape[0]), order='F')       
             
    return x
#=======================================================================

#=======================================================================
def get_filename(X):
    # Get NIfTI filename
    #___________________________________________________________________    
    
    N = len(X)    
    
    if N == 1:
        f = X.get_filename()
    else: 
        f = ''
        for n in range(0,N):
            f = f + '\n' + X[n].get_filename()
        
    return f
#=======================================================================    
