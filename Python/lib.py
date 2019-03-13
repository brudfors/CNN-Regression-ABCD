#---------------------------------------------------------------------------
# Utility functions used in the ABCD Neurocognitive Prediction Challenge 
# (ABCD-NP-Challenge 2019) 
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# All imports here
#---------------------------------------------------------------------------

from keras.layers import Flatten, Dense, Dropout, ZeroPadding3D, Conv3D, Activation, MaxPooling3D,\
    GlobalAveragePooling3D, Average, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2, l1
import nibabel as nib
import random
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras import callbacks
from keras.models import load_model
from keras.optimizers import Adam
import copy 

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
    # Good ref on CNN regression: https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/
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
def get_model_new(Shape,Do2D = False,filters=([64,2],[128,2], [256,2]),dense=([500,0.5],[100,0.25],[20,0]),batchNorm=True):
    # Returns a Keras regression CNN model. 
    # Good ref on CNN regression: https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/
    #___________________________________________________________________]

    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = Shape
    chanDim    = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs      

        # CONV => RELU => BN => POOL
        for i1 in range(0,f[1]):
            if Do2D:
                x = Conv2D(f[0], 3, padding="same", kernel_initializer = 'he_normal')(x)            
            else:
                x = Conv3D(f[0], 3, padding="same", kernel_initializer = 'he_normal')(x)
            x = Activation("relu")(x)
        
        if batchNorm:            
            x = BatchNormalization(axis=chanDim)(x)
            
        if Do2D:
            x = MaxPooling2D(pool_size=(2, 2))(x)
        else:       
            x = MaxPooling3D(pool_size=(2, 2, 2))(x)
	    
    # Flatten the volume...
    x = Flatten()(x)

    # Then do repetions of FC => RELU => BN => DROPOUT
    if isinstance(dense, tuple):
        for (i, f) in enumerate(dense):
            x = Dense(f[0])(x)
            x = Activation("relu")(x)
            
            if batchNorm:
                x = BatchNormalization(axis=chanDim)(x)
                
            if f[1] > 0:
                x = Dropout(f[1])(x)
    else:
        x = Dense(dense[0])(x)
        x = Activation("relu")(x)

        if batchNorm:
            x = BatchNormalization(axis=chanDim)(x)
            
        if dense[1] > 0:
            x = Dropout(dense[1])(x)  
            
    # Add the regression node
    x = Dense(1, activation="linear")(x)
	    
    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
#=======================================================================

#=======================================================================
def data_generator(X, y, Shape, BatchSize=8, Do2D=False):
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
               
            img = get_img(X[i],Shape,Do2D)
            
            x1[batch_idx % BatchSize] = img
            y1[batch_idx % BatchSize] = y[i]
            batch_idx += 1
            
            if (batch_idx % BatchSize) == 0:
                yield (x1, y1)
#=======================================================================       

#=======================================================================
def get_abcd_training_data_img(Dir,Scores,S=0):
    # Returns training data for the ABCD challenge, where Dir is a 
    # directory of NIfTIs, and Scores is a dictionary that contains the
    # residual IQ scores.
    #___________________________________________________________________
        
    X   = []
    y   = []
    cnt = 0
    for root, directories, filenames in os.walk(Dir):
        for filename in filenames:                         
            ix   = filename.find('NDAR')
            Name = filename[ix:ix + 16]

            # Targets
            y.append(Scores[Name])
            
            # Images
            X.append(nib.load(os.path.join(root,filename)))
            
            cnt += 1                
            if cnt > 0 and cnt >= S:
                break  
                    
    Shape = X[0].header.get_data_shape()
    Shape = (Shape[0],Shape[1],Shape[2],1)
            
    return X,y,Shape
#=======================================================================

#=======================================================================
def get_abcd_training_data_seg(Dir,Scores,S=0):
    # Returns training data for the ABCD challenge, where Dir is a 
    # directory of directories, where each inner directory contains a 
    # set of NIfTIs (e.g., segmentations), and Scores is a dictionary 
    # that contains the residual IQ scores.
    #___________________________________________________________________
           
    X   = []
    y   = []
    cnt = 0
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
                
                cnt += 1                
                if cnt > 0 and cnt >= S:
                    break                     
    
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
def get_nii_data(nii,Do2D=False):
    # More memory efficient way of getting NIfTI data, see:
    # https://nipy.org/nibabel/images_and_memory.html
    #___________________________________________________________________
    
    if Do2D:
        dm  = nii.header.get_data_shape()
        z   = int(np.round(dm[2]/2))
        img = np.asarray(nii.dataobj[..., z])
        img = copy.deepcopy(img)
        return img 
    else:
        return np.asarray(nii.dataobj)
#=======================================================================

#=======================================================================
def get_img(X,Shape,Do2D=False):
    # Get a, possibly, multi-channel image
    #___________________________________________________________________

    if Shape[-1] > 1:
        # Multi-channel data
        img = np.zeros(Shape, dtype=np.float32)
        for i1 in range(0,len(X)):
            if Do2D:
                img[:,:,i1]   = get_nii_data(X[i1],Do2D)
            else:
                img[:,:,:,i1] = get_nii_data(X[i1],Do2D)
    else:
        # Single-channel data
        img = get_nii_data(X,Do2D)
        img = np.expand_dims(img, axis=-1)
        
    #img[np.isnan(img)] = 0
    #img[img < 0]       = 0
    #img                = img / img.max()
    
    return img
#=======================================================================

#=======================================================================        
def imshow(X,Shape):
    # Show a bunch of images
    #___________________________________________________________________

    plt.figure(1)
    if Shape[-1] == 1:
        # Single plot
        x = get_nii_data(X,Do2D=True)
        plt.imshow(x)
        plt.axis('off') 
    else:
        x = np.zeros(Shape, dtype=np.float32)
        for i1 in range(0,Shape[-1]):
            x[:,:,i1] = get_nii_data(X[i1],Do2D=True)
        plt.imshow(x)
        plt.axis('off')             
    
    plt.colorbar()
    plt.show()    
#=======================================================================    

#=======================================================================
def get_slice(X,Shape):
    # Get slice(s) in the z-direction
    #___________________________________________________________________    
    
    if isinstance(X, (list,)):
        N  = len(X)     
    else:
        N = 1
        
    x = get_img(X,Shape,True)
    print(x.shape)
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
    
    if isinstance(X, (list,)):
        N  = len(X)     
    else:
        N = 1 
    
    if N == 1:
        f = X.get_filename()
    else: 
        f = ''
        for n in range(0,N):
            f = f + '\n' + X[n].get_filename()
        
    return f
#=======================================================================

#=======================================================================
def get_names_abcd(Dir):
    # Separate ABCD subject names into training and validation partitions
    #___________________________________________________________________  
    
    NamTrain = {}
    NamVal   = {}
    for R0, D0, F0 in os.walk(Dir):
        for d0 in D0:        
            for R1, D1, F1 in os.walk(os.path.join(R0,d0)):
                for f1 in F1:            
                    Name = f1.split("_")   
                    Name = Name[0]
                    if d0 == 'training':                        
                            NamTrain[Name] = True
                    elif d0 == 'validation':
                            NamVal[Name] = True    
                            
    return NamTrain, NamVal                         
#=======================================================================

#=======================================================================
def X_split_abcd(Dir,ScoresTrain,ScoresVal,Do2D):
    # Split ABCD data into train, val and test sets
    #___________________________________________________________________  
            
    X_train = []
    X_valid = []
    X_test  = []
    y_train = []
    y_valid = []
    Shape   = []
    
    for item in os.listdir(Dir):
        
        if os.path.isfile(os.path.join(Dir,item)):
            # Single-channel
            ix   = item.find('NDAR')
            Name = item[ix:ix + 16]

            if Name in ScoresTrain:
                y_train.append(ScoresTrain[Name])                
                X_train.append(nib.load(os.path.join(Dir,item)))                                
            elif Name in ScoresVal:
                y_valid.append(ScoresVal[Name])                
                X_valid.append(nib.load(os.path.join(Dir,item)))              
            else:           
                X_test.append(nib.load(os.path.join(Dir,item)))   
                                                       
        else:
            # Multi-channel            
            X1 = []
            for f in os.listdir(os.path.join(Dir,item)):   
                X1.append(nib.load(os.path.join(Dir,item,f)))
                    
            ix   = f.find('NDAR')
            Name = f[ix:ix + 16]

            if Name in ScoresTrain:
                y_train.append(ScoresTrain[Name])                
                X_train.append(X1)                                
            elif Name in ScoresVal:
                y_valid.append(ScoresVal[Name])                
                X_valid.append(X1)              
            else:           
                X_test.append(X1)   
                        
    if os.path.isfile(os.path.join(Dir,item)):
        # Single-channel                
        Shape = X_train[0].header.get_data_shape()
        if Do2D:
            Shape = (Shape[0],Shape[1],1)
        else:
            Shape = (Shape[0],Shape[1],Shape[2],1)
    else:
        # Multi-channel
        Shape = X_train[0][0].header.get_data_shape()
        if Do2D:
            Shape = (Shape[0],Shape[1],len(X_train[0]))                                   
        else:
            Shape = (Shape[0],Shape[1],Shape[2],len(X_train[0]))                                   
        
    # Convert targets to Numpy arrays
    y_train = np.asarray(y_train).astype(np.float)       
    y_valid = np.asarray(y_valid).astype(np.float)
           
    return X_train, X_valid, X_test, y_train, y_valid, Shape
#=======================================================================

#=======================================================================
def run_predict(PthTrainScr,PthValScr,DirData,PthModel,NbEpochs,BatchSize):
    # Predict on a dataset
    #___________________________________________________________________  
    
    # Get ABCD targets (IQ scores)
    ScoresTrain = get_abcd_scores(PthTrainScr)
    ScoresVal   = get_abcd_scores(PthValScr)    
    
    # Get training data (X,y)
    X_train, X_valid, X_test, y_train, y_valid, Shape  = X_split_abcd(DirData,ScoresTrain,ScoresVal)
    
    TotalTrainingSamples   = len(X_train)
    TotalValidationSamples = len(X_valid)
        
    # Define generators
    training_generator   = data_generator(X_train,y_train,Shape,BatchSize)
    validation_generator = data_generator(X_valid,y_valid,Shape,BatchSize)
    
    # Define CNN regression model
    model = get_model_new(Shape)
    
    # Compile CNN model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=['mse'])

    # Define callback to ensure we save the best performing model
    checkpointer = callbacks.ModelCheckpoint(filepath=PthModel, 
                                             verbose=0, 
                                             save_best_only=True)    

    # Fit model
    # For training loss, keras does a running average over the batches. 
    # For validation loss, a conventional average over all the batches in validation data is performed. 
    model.fit_generator(training_generator,
                        nb_epoch=NbEpochs,
                        validation_data=validation_generator,
                        steps_per_epoch=TotalTrainingSamples/BatchSize,
                        validation_steps=TotalValidationSamples/BatchSize,
                        callbacks=[checkpointer], # callbacks=[tbCallBack]
                        verbose=0) # 0 = silent, 1 = progress bar, 2 = one line per epoch

    # Load best model
    del model 
    model = load_model(PthModel)
    
    yp  = model.predict_generator(validation_generator,steps=TotalValidationSamples)
    mse = np.mean(np.square(y_valid - yp))
    print('MSE = ' + str(mse))
#=======================================================================    
