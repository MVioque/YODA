#Young Object Discoverer Algorithm (YODA)

from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import regularizers
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from keras.utils import plot_model
import matplotlib.backends.backend_pdf 
from keras import regularizers
from astropy.io import fits
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn import preprocessing
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn import preprocessing
from sklearn.utils import resample
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from google.colab import files
from google.colab import drive
import keras_metrics as km
import csv
import os
import tempfile
from keras.callbacks import Callback

#Measure duration of algorithm and plots elapsed time at the end
start_time = time.time()

#Load Input sample.
#Load the whole set to consider with the characteristics to use including the category (label). 
FINAL_COMPLETE = np.loadtxt(open("/content/gdrive/My Drive/Colab Notebooks/FCT_no_i_no_rep3.csv"), delimiter=',', skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))

#We load the special sources with the characteristics to use including the category (label). 
Special_sources = np.loadtxt(open("/content/gdrive/My Drive/Colab Notebooks/Special_sources_all_info3.csv"), delimiter=',', skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))

PMS_number = 848 #PMS sources in Special sources (can be counted by hand)
COMPLETE_number = 4150983 #Number with no repetitions

#Hyper-parameters to choose:
ite_num = 30 #bootsrapped samples to consider
NNeurons = 580 #Number of neurons per layer (layers of different neurons are possible, but need to be individually specified)
rate = 0.5 #Dropout rate (by default, dropout is applyied to all layers)
Prop = 1-(PMS_number/(0.0018*COMPLETE_number))  #Proportion of Input sample sources to be used in the training set (= to the proportion of PMS sources in Gaia)
Test_s = 0.1 #Choose test set size
Cross_s = 0.1 #Choose Cross-Validadtion set size
Probability_s = 0.50 #Probability threshold from which to select PMS objects:
fig, ax = plt.subplots()

Precision_list50 = []
Recall_list50 = []
TP_list50 = []
FN_list50 = []
FP_list50 = []
TN_list50 = []

Precision_list50Be = []
Recall_list50Be = []
TP_list50Be = []
FN_list50Be = []
FP_list50Be = []
TN_list50Be = []

ite = 0

while ite <ite_num: #Start bootstrapping runs
  
    print('Bootstraped iteration:', ite)
    #From the whole set, we seperate a random number of sources equal to the porportion of PMS sources.
    Training_others, Input_set = train_test_split(FINAL_COMPLETE, test_size=Prop)

    #Bootstrap the special sources
    Special_sources_boot = resample(Special_sources, replace = True, n_samples=len(Special_sources), random_state=None)

    #Create input set by concatenation.
    Input_for_training = np.concatenate((Training_others,Special_sources_boot))

    print('Input_for_training =', Input_for_training.shape[0], '   100%')


    #Select random elements from the imput set. 80% training set, 20% test set (and they get shuffled).
    Training_set, Test_set = train_test_split(Input_for_training, test_size=Test_s)                              
    
    #Load train and test set
    x_train = Training_set[:,[3,4,5,6,7,8,9,10,11,12,13,14,15]]
    y_train = Training_set[:,16]

    x_test = Test_set[:,[3,4,5,6,7,8,9,10,11,12,13,14,15]]
    y_test = Test_set[:,16]    
         
    #Create all combinations of colours, but do not combine wih the first three characteristics that are not passbands.
    New_array = np.c_[x_train[:,0]]
    New_array = np.c_[New_array, x_train[:,1]]
    New_array = np.c_[New_array, x_train[:,2]]

    a = 0
    b= 0
    while a+3 < x_train.shape[1]:
        b = 0
        while b+4+a < x_train.shape[1]:   
            New_col = x_train[:,a+3]-x_train[:,b+4+a]
            New_array = np.c_[New_array, New_col]
            b = b+1
        a = a+1

    scaler = preprocessing.StandardScaler().fit(New_array)
    New_array = scaler.transform(New_array)

    # Run PCA on this dataset
    pca = PCA(n_components=New_array.shape[1])
    pca.fit(New_array)

    # extract the variance of each of the components
    variance = pca.explained_variance_

    # extract the components, not necessary to print them
    components = pca.components_
    #print("components: ", components)

    #plt.figure(1)
    #plt.yscale('log')
    Total_var = np.sum(variance)

    #plt.plot(np.arange(1, New_array.shape[1]+1), (variance/Total_var)*100,'ko',ms=3,mew=3)
    #plt.xlabel('Principal components', fontsize=15)
    #plt.ylabel('Variance (%)', fontsize=15)

    Total_var = np.sum(variance)
 
    # We will now examine how many dimensions do we want by only chooseing those that retain 99.99% of the variance.
    variance_list =[]
    for i in range(len(variance)):
        variance_list.append(variance[i])
        if variance[i+1]/variance[i]<=0.01:
            print(round((np.sum(variance_list)/np.sum(variance))*100,2),'of the variance is retained with',i+1,'components')
            break

    Components_chosen = i+1
    
    #Apply chosen PCA to number of dimensions
    pca = PCA(n_components=Components_chosen)
    pca.fit(New_array)
    New_array_reduced = pca.transform(New_array) 
    print('')
    print("Shape of original Training Set: ", New_array.shape)
    print("Shape of reduced Training Set: ",New_array_reduced.shape)
    
    #Number of layers, THIS IS NOT AN HYPER-PARAMETER, if less of more layers are needed they need to be added in Keras manually
    Layers = 2
    
    #Linearization of the categories
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)
   
    #As it is a very skewed problem, we will used batches of the size of the Training Set.
    batch_size_value = len(New_array_reduced)

    #Compute class weights, not is is balanced but other class weights can be applied.
    Weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train),
                                                     y_train)

    class_weights = {0:Weights[0],
                    1: Weights[1],
                    2: Weights[2]}
    
    #Neural Network
    if Layers == 2:
            model = Sequential()
            model.add(Dense(units=int(NNeurons), activation='relu', input_dim=Components_chosen,kernel_regularizer=regularizers.l2(0.01)))
            model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
            model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
            model.add(Dense(units=int(NNeurons), activation='relu', input_dim=int(NNeurons),kernel_regularizer=regularizers.l2(0.01)))
            model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
            model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))     
            model.add(Dense(units=3, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
            model.compile(loss='categorical_crossentropy',
                           optimizer='adamax', #How to update de learning rate. categorical_crossentropy
                           metrics=['accuracy',km.categorical_precision(label=1), km.categorical_recall(label=1)])  
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_precision', min_delta=0, patience=50, verbose=1, mode='max', baseline=None)#
            history = model.fit(New_array_reduced, one_hot_labels, epochs=3000,validation_split=Cross_s, batch_size=batch_size_value, verbose=0,class_weight=class_weights, callbacks= [earlystopping])
            
            #Optionally, cost functions, accuracy, precision and recall of cross-validation set can be plotted.
            #plt.figure(2)
            #plt.plot(history.history['loss']), plt.plot(history.history['val_loss'])
            #plt.ylabel('Cost function', fontsize = 12)
            #plt.xlabel('Iterations', fontsize = 12)
            #plt.figure(3)
            #plt.plot(history.history['acc']), plt.plot(history.history['val_acc'])
            #plt.ylabel('Accuracy', fontsize = 12)
            #plt.xlabel('Iterations', fontsize = 12)        
            #plt.figure(4)
            #plt.plot(history.history['precision']), plt.plot(history.history['val_precision'])
            #plt.ylabel('Precision', fontsize = 12)
            #plt.xlabel('Iterations', fontsize = 12)
            #plt.figure(5)
            #plt.plot(history.history['recall']), plt.plot(history.history['val_recall'])
            #plt.ylabel('Recall', fontsize = 12)
            #plt.xlabel('Iterations', fontsize = 12)  
            #print('Precision = ', history.history['val_precision'][-1])
            #print('Recall = ', history.history['val_recall'][-1])      
            #plt.clf()
    
    print()
    print('RESULTS ON TEST SET')
    #CHOOSEN MODEL ON TEST SET
    #Important to scale as done for the training set first or transform to PCA
    New_array_test = np.c_[x_test[:,0]]
    New_array_test = np.c_[New_array_test, x_test[:,1]]
    New_array_test = np.c_[New_array_test, x_test[:,2]]

    #Create all combinations of colours
    a = 0
    b= 0
    while a+3 < x_test.shape[1]:
        b = 0
        while b+4+a < x_test.shape[1]:   
            New_col = x_test[:,a+3]-x_test[:,b+4+a]
            New_array_test = np.c_[New_array_test, New_col]
            b = b+1
        a = a+1

    New_array_test = scaler.transform(New_array_test)
    New_array_reduced_test = pca.transform(New_array_test) 

    test_classes = model.predict(New_array_reduced_test)
    PMS_Chance = test_classes[:,1]
    Be_Chance = test_classes[:,2]
    Other_Chance = test_classes[:,0]

    countTP = 0
    countFN = 0
    countFP = 0
    countTN = 0

    #Get precision and recall (other metrics, like e.g., F1 score can be added):

    #Value from which to select PMS objects:
    Probability = Probability_s
    print('Probability threshold =',Probability)

    for i in range(len(New_array_reduced_test)):
       if PMS_Chance[i]>=Probability and y_test[i]==1:
            countTP  = countTP + 1
       if PMS_Chance[i]<Probability and y_test[i]==1:
            countFN  = countFN + 1
       if PMS_Chance[i]>=Probability and y_test[i]!=1:
            countFP  = countFP + 1
       if PMS_Chance[i]<Probability and y_test[i]!=1:
            countTN  = countTN + 1 

    print('Number of PMS objects in test set =', countTP+countFN)     

    #Precision: Of all objects for which we have predicted a PMS nature, what fraction is actually a PMS?
    Precision = countTP/(countTP+countFP)

    #Recall: Of all obejcts that are actually of PMS nature, what fraction have we detected as PMS?
    Recall = countTP/(countTP+countFN)
      
    Precision_list50.append(Precision)
    Recall_list50.append(Recall)
    TP_list50.append(countTP)
    FN_list50.append(countFN)
    FP_list50.append(countFP)
    TN_list50.append(countTN)
     
    print('Precision=',Precision_list50)
    print('Recall=',Recall_list50)
    print('TP=',TP_list50)
    print('FN=',FN_list50)
    print('FP=',FP_list50)
    print('TN=',TN_list50)
    
    countTP = 0
    countFN = 0
    countFP = 0
    countTN = 0
    
    for i in range(len(New_array_reduced_test)):
       if Be_Chance[i]>=Probability and y_test[i]==2:
            countTP  = countTP + 1
       if Be_Chance[i]<Probability and y_test[i]==2:
            countFN  = countFN + 1
       if Be_Chance[i]>=Probability and y_test[i]!=2:
            countFP  = countFP + 1
       if Be_Chance[i]<Probability and y_test[i]!=2:
            countTN  = countTN + 1 

    print('Number of PMS objects in test set =', countTP+countFN)     

    #Precision: Of all objects for which we have predicted a PMS nature, what fraction is actually a PMS?
    PrecisionBe = countTP/(countTP+countFP)

    #Recall: Of all obejcts that are actually of PMS nature, what fraction have we detected as PMS?
    RecallBe = countTP/(countTP+countFN)
      
    Precision_list50Be.append(PrecisionBe)
    Recall_list50Be.append(RecallBe)
    TP_list50Be.append(countTP)
    FN_list50Be.append(countFN)
    FP_list50Be.append(countFP)
    TN_list50Be.append(countTN)
     
    print('PrecisionBe=',Precision_list50Be)
    print('RecallBe=',Recall_list50Be)
    print('TPBe=',TP_list50Be)
    print('FNBe=',FN_list50Be)
    print('FPBe=',FP_list50Be)
    print('TNBe=',TN_list50Be)
    
    
    #Recall vs. Precision plot
    import matplotlib.backends.backend_pdf 

    pdf = matplotlib.backends.backend_pdf.PdfPages("Precision_vs_recall.pdf")

    countTP_PMS = 0
    countFN_PMS = 0
    countFP_PMS = 0
    countTN_PMS = 0
    Precision_list_PMS = []
    Recall_list_PMS = []
    countTP_Be = 0
    countFN_Be = 0
    countFP_Be = 0
    countTN_Be = 0
    Precision_list_Be = []
    Recall_list_Be = []
    Count_FP_Be = []
    Count_TP_Be = []

    for P in np.linspace(0, 1, 101): 
     countTP_PMS = 0
     countFN_PMS = 0
     countFP_PMS = 0
     countTN_PMS = 0
     countTP_Be = 0
     countFN_Be = 0
     countFP_Be = 0
     countTN_Be = 0 
     for i in range(len(x_test)):
       if PMS_Chance[i]>=P and y_test[i]==1:
            countTP_PMS  = countTP_PMS + 1
       if PMS_Chance[i]<P and y_test[i]==1:
            countFN_PMS  = countFN_PMS + 1
       if PMS_Chance[i]>=P and y_test[i]!=1:
            countFP_PMS  = countFP_PMS + 1
       if PMS_Chance[i]<P and y_test[i]!=1:
            countTN_PMS  = countTN_PMS + 1
       if Be_Chance[i]>=P and y_test[i]==2:
            countTP_Be  = countTP_Be + 1
       if Be_Chance[i]<P and y_test[i]==2:
            countFN_Be  = countFN_Be + 1
       if Be_Chance[i]>=P and y_test[i]!=2:
            countFP_Be  = countFP_Be + 1
       if Be_Chance[i]<P and y_test[i]!=2:
            countTN_Be  = countTN_Be + 1 
     Count_FP_Be.append(countFP_Be)
     Count_TP_Be.append(countTP_Be) 
     if countTP_PMS+countFP_PMS == 0 or countTP_PMS+countFN_PMS ==0 or countTP_PMS==0:
       Precision_list_PMS.append(None)
       Recall_list_PMS.append(None)
     else:    
       Precision_list_PMS.append(countTP_PMS/(countTP_PMS+countFP_PMS))
       Recall_list_PMS.append(countTP_PMS/(countTP_PMS+countFN_PMS)) 
     if countTP_Be+countFP_Be == 0 or countTP_Be+countFN_Be ==0 or countTP_Be==0:   
          Precision_list_Be.append(None)
          Recall_list_Be.append(None)
     else:    
       Precision_list_Be.append(countTP_Be/(countTP_Be+countFP_Be))
       Recall_list_Be.append(countTP_Be/(countTP_Be+countFN_Be)) 
     if P==0.25:
      Precision25 = countTP_PMS/(countTP_PMS+countFP_PMS)
      Recall25 = countTP_PMS/(countTP_PMS+countFN_PMS)
      PrecisionBe25 = countTP_Be/(countTP_Be+countFP_Be)
      RecallBe25 = countTP_Be/(countTP_Be+countFN_Be)
     if P==0.75:
      Precision75 = countTP_PMS/(countTP_PMS+countFP_PMS)
      Recall75 = countTP_PMS/(countTP_PMS+countFN_PMS)
      PrecisionBe75 = countTP_Be/(countTP_Be+countFP_Be)
      RecallBe75 = countTP_Be/(countTP_Be+countFN_Be)
      
    plt.plot(Recall, Precision,'or',ms=7,markeredgecolor='black',markeredgewidth=1.5, zorder=1)
    plt.plot(Recall25, Precision25,'vr',ms=7,markeredgecolor='black',markeredgewidth=1.5, zorder=1)
    plt.plot(Recall75, Precision75,'^r',ms=7, markeredgecolor='black',markeredgewidth=1.5,zorder=1)    
    plt.plot(Recall_list_PMS, Precision_list_PMS, zorder=-1,color='blue')
    plt.plot(RecallBe, PrecisionBe,'og',ms=7,markeredgecolor='black',markeredgewidth=1.5, zorder=1)
    plt.plot(RecallBe25, PrecisionBe25,'vg',ms=7,markeredgecolor='black',markeredgewidth=1.5, zorder=1)
    plt.plot(RecallBe75, PrecisionBe75,'^g',ms=7,markeredgecolor='black',markeredgewidth=1.5, zorder=1)    
    plt.plot(Recall_list_Be, Precision_list_Be, zorder=-1,color='gray')      
    plt.ylabel('Precision', fontsize = 15)
    plt.xlabel('Recall', fontsize = 15)

    ax.tick_params(reset='True', axis='both', which='both', direction='out')

    majorLocator = MultipleLocator(0.1)
    majorFormatter = FormatStrFormatter('%1.1f')
    minorLocator = MultipleLocator(0.05)

    majorLocator1 = MultipleLocator(0.1)
    majorFormatter1 = FormatStrFormatter('%1.1f')
    minorLocator1 = MultipleLocator(0.05)

    ax.xaxis.set_major_locator(majorLocator1)
    ax.xaxis.set_major_formatter(majorFormatter1)
    ax.xaxis.set_minor_locator(minorLocator1)

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)


    print('Number of PMS objects in test set =', countTP_PMS+countFN_PMS)     
    print('Number of Be objects in test set =', countTP_Be+countFN_Be)     
    print(countTP_PMS)
    print(countFP_PMS)
    print(countTP_Be)
    print(Count_FP_Be)
    print(Count_TP_Be)

#        
    x_sample = Input_set[:,[3,4,5,6,7,8,9,10,11,12,13,14,15]]

    #Important to scale as done for the training set first or transform to PCA
    New_array_sample = np.c_[x_sample[:,0]]
    New_array_sample = np.c_[New_array_sample, x_sample[:,1]]
    New_array_sample = np.c_[New_array_sample, x_sample[:,2]]


    #Create all combinations of colours
    a = 0
    b= 0
    while a+3 < x_sample.shape[1]:
        b = 0
        while b+4+a < x_sample.shape[1]:   
            New_col = x_sample[:,a+3]-x_sample[:,b+4+a]
            New_array_sample = np.c_[New_array_sample, New_col]
            b = b+1
        a = a+1

    New_array_sample = scaler.transform(New_array_sample)
    New_array_reduced_sample = pca.transform(New_array_sample) 

    Final_classes = model.predict(New_array_reduced_sample)
    PMS_Chance = Final_classes[:,1]
    Be_Chance = Final_classes[:,2]
    Other_Chance = Final_classes[:,0]
    Output_final_set = np.append(Input_set, Final_classes, axis=1)
    np.savetxt("Output_Input_sample_v3_no_r_Ha{}.csv".format(ite), Output_final_set, delimiter=",")
    #files.upload("Output_Input_sample_no_Var{}.csv".format(ite))
    uploaded = drive2.CreateFile({'title': 'Output_Input_sample_v3_no_r_Ha{}.csv'.format(ite)})
    uploaded.SetContentFile('Output_Input_sample_v3_no_r_Ha{}.csv'.format(ite))
    uploaded.Upload()
    #files.download("Output_Input_sample{}.csv".format(ite))  
    
    
    ite = ite+1

print("--- %s minutes ---" % ((time.time() - start_time)/60))
pdf.savefig()
pdf.close()
