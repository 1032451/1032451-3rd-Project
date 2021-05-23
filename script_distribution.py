#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras import backend as K
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RandomizedSearchCV
from keras.utils.layer_utils import count_params

from numpy.random import seed
my_seed=1
seed(my_seed)
np.random.seed(my_seed)

import random
random.seed(my_seed)

tf.random.set_seed(my_seed)

from tensorboard.plugins.hparams import api as hp

#------------------------------------------------------------------------------

rng = np.random.default_rng(my_seed)

kfold = KFold(n_splits = 10, shuffle=False)

def load_features(file_name):
    # dictionary to load features into
    features = {}

    # load file using the HDF5 library
    f = h5py.File(file_name, 'r')

    print("Loading features from {0} ... ".format(file_name))

    # loop over each feature contained in the file
    for feature_name in f.keys():
        # convert to numpy format and store in dictionary
        x = np.array(f[feature_name])
        print(feature_name, "{0}x{1}".format(x.shape[0], x.shape[1]))
        features[feature_name] = x

    return features

def load_annotations(file_name):
    with open(file_name) as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i].split(" ")
        content[i] = [int(content[i][0]),int(content[i][1])]
    #print(content) 
    return content

#------------------------------------------------------------------------------

#video_features = load_features('Hollywood-dev/features/SavingPrivateRyan_visual.mat')
#annotations = load_annotations('Hollywood-dev/annotations/SavingPrivateRyan_blood.txt')
#no_frames = len(video_features["CM"])

#violent=[0]*(no_frames)

#print(train_features)

HP_NUM_TRIAL = hp.HParam('num_trial',hp.RealInterval(0.0,100.0))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512,2048,8192]))
HP_REGULARIZATION = hp.HParam('regularization_weight', hp.RealInterval(-5.0,-1.0))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([0,1,2,3]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(-4.0,-1.0))
HP_REGULARIZE_FUNCTION = hp.HParam('regularization_function',hp.Discrete(['l1','l2']))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0,0.8))
HP_LAYER_WIDTH = hp.HParam('hidden layer width', hp.Discrete([32,64,128,256,512,1024]))

ACC = 'accuracy'
TP = 'tp'
TN = 'tn'
FN = 'fn'
FP = 'fp'
ACC_VAL = 'accuracy_val'
TP_VAL = 'tp_val'
TN_VAL = 'tn_val'
FN_VAL = 'fn_val'
FP_VAL = 'fp_val'
RECALL = 'recall'
RECALL_VAL = 'recall_val'
PRECISION = 'precision'
PRECISION_VAL = 'precision_val'
F1 = 'f1'
F1_VAL = 'f1_val'
TRAIN = 'train'

num_trials = 10

hparam_folder = 'logs/distribution'
threshold_file = 'thresholds.txt' 

with tf.summary.create_file_writer(hparam_folder).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_TRIAL,HP_NUM_UNITS, HP_REGULARIZATION, HP_OPTIMIZER, HP_NUM_LAYERS, HP_LEARNING_RATE, HP_REGULARIZE_FUNCTION, HP_DROPOUT, HP_LAYER_WIDTH],
    metrics=[hp.Metric(TRAIN, display_name='Number of parameters'),
            hp.Metric(ACC, display_name='Accuracy'),hp.Metric(ACC_VAL, display_name='Validation Accuracy'),
            hp.Metric(RECALL, display_name='Recall'),hp.Metric(RECALL_VAL, display_name='Validation Recall'),
            hp.Metric(PRECISION, display_name='Precision'),hp.Metric(PRECISION_VAL, display_name='Validation Precision'),
            hp.Metric(F1, display_name='F1 Score'),hp.Metric(F1_VAL, display_name='Validation F1 Score'),
            hp.Metric(TP, display_name='True Positives'),hp.Metric(FP, display_name='False Positives'),
            hp.Metric(TN, display_name='True Negatives'),hp.Metric(FN, display_name='False Negatives'),
            hp.Metric(TP_VAL, display_name='Validation True Positives'),hp.Metric(FP_VAL, display_name='Validation False Positives'),
            hp.Metric(TN_VAL, display_name='Validation True Negatives'),hp.Metric(FN_VAL, display_name='Validation False Negatives')
           ],
  )


my_validation_split = 0.1
using_val_data = True
using_normalisation = True
my_hidden_layers = [32]
my_epochs = 20
my_batch_size = 512

#with open("file.csv","w") as file:
#    file.write("val\n")
#    for x in range(1,10**8):
#        file.write(str(x)+"\n")
#train=pd.read_csv("file.csv",chunksize=10**6)
#s = 0
#for chunk in train:
#    print(chunk.iloc[0][0])
#sys.exit()

def read_data():
    train = pd.read_csv("dataset_train_small.csv")

    train_features = train.copy()
    train_features = train_features.dropna()
    train_features = train_features.sample(frac=1,random_state=my_seed)
    #train_features = train_features[:1000]
    train_labels = train_features.pop('V')
    #train_labels = train_labels.sample(frac=1)

    validation = pd.read_csv("dataset_test_small.csv")
    validation_features = validation.copy()
    validation_features = validation_features.dropna()
    validation_features = validation_features.sample(frac=1,random_state=my_seed)
    #validation_features = validation_features[:1000]
    validation_labels = validation_features.pop('V')
    
    return train_features,train_labels,validation_features,validation_labels

def read_data_as_chunks():
    train = pd.read_csv("dataset_train_full.csv", chunksize=10**6)
    return train

def class_weights():
    v = len([x for x in train_labels if x==1])
    nv = len([x for x in train_labels if x==0])
    return {1:1.0,0:v/nv}

def make_title():
    title = ""
    if using_val_data:
        title+="Using validation set from different movies, "
    else:
        title+="Using "+str(my_validation_split)+" of the data for validation, "
    if using_normalisation:
        title+=" with normalisation, "
    else:
        title+=" without normalisation, "
    title+="hidden layers: "+str(my_hidden_layers)+", "
    title+="batch size: "+str(my_batch_size)+", "
    title+="epochs: "+str(my_epochs)
    return title    

def visualize(history, title, metric):
    loss = history.history[metric]
    val_loss = history.history["val_"+metric]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training "+metric)
    plt.plot(epochs, val_loss, "r", label="Validation "+metric)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/"+"".join(title.split(':'))+" "+metric+".png",bbox_inches='tight')

def build_model(hparams):
    my_hidden_layers = [hparams[HP_LAYER_WIDTH]]*hparams[HP_NUM_LAYERS]
    model = tf.keras.Sequential()
    if using_normalisation:
        model.add(layers.BatchNormalization())
    for h in my_hidden_layers:
        kr = None
        if hparams[HP_REGULARIZE_FUNCTION]=='l1':
            kr = l1(hparams[HP_REGULARIZATION])
        elif hparams[HP_REGULARIZE_FUNCTION]=='l2':
            kr = l2(hparams[HP_REGULARIZATION])
        else:
            Error("Bad regularisation function")
        model.add(layers.Dense(h,activation='relu',kernel_regularizer=kr))
        model.add(layers.Dropout(hparams[HP_DROPOUT],seed=my_seed))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model

train_features,train_labels,validation_features,validation_labels=read_data()

def run_progressive_loading(logdir,hparams):
    with tf.summary.create_file_writer(logdir).as_default():
        model = build_model(hparams)
        model.compile(loss = tf.losses.BinaryCrossentropy(), optimizer = tf.optimizers.Adam(), metrics=['accuracy',tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

        train = pd.read_csv("dataset_train_full.csv", chunksize=10**5)
        for chunk in train:
            train_features = chunk.dropna()
            train_features = train_features(frac=1, random_state=my_seed)
            train_labels = train_features.pop("V")
            
            model.fit(
            train_features, train_labels, epochs=my_epochs, batch_size=my_batch_size, 
            validation_split=0.1,
            class_weight=class_weights(),
            callbacks=[
                tf.keras.callbacks.TensorBoard(logdir+"full", histogram_freq=1),  # log metrics
                hp.KerasCallback(logdir, hparams),  # log hparams
                ],
            verbose=1,
            )


#print("Violent: "+str(100.0*len([x for x in train_labels if x==1])/len(train_labels)))

loss,acc,prec,rec=[],[],[],[]

def kfoldrun():
    for train, test in kfold.split(train_features,train_labels):
            hparams={
                HP_NUM_LAYERS:1,
                HP_NUM_UNITS:5120,
                HP_OPTIMIZER:'adam',
                HP_REGULARIZATION:0.0001
                }
    
            model = build_model(hparams)
        
            model.compile(loss = tf.losses.BinaryCrossentropy(), optimizer = tf.optimizers.Adam(), metrics=['accuracy',tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
        
            trf,trl,vlf,vll=train_features.iloc[train], train_labels.iloc[train], train_features.iloc[test], train_labels.iloc[test]
            history = model.fit(
            trf, trl, epochs=my_epochs, batch_size=my_batch_size, 
            validation_data=(vlf,vll),
            class_weight=class_weights(),
            callbacks=[
                #tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                #hp.KerasCallback(logdir, hparams),  # log hparams
                ],
                verbose=1,
            )
        
            loss.append(history.history['val_loss'][-1])
            acc.append(history.history['val_accuracy'][-1])
            prec.append(history.history['val_precision'][-1])
            rec.append(history.history['val_recall'][-1])

    print(sum(loss)/len(loss))
    print(sum(acc)/len(acc))
    print(sum(prec)/len(prec))
    print(sum(rec)/len(rec))


def run(logdir,hparams):
    with tf.summary.create_file_writer(logdir).as_default():
        model = build_model(hparams)
        optimizer = None
        if hparams[HP_OPTIMIZER] == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate = hparams[HP_LEARNING_RATE])
        elif hparams[HP_OPTIMIZER] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate = hparams[HP_LEARNING_RATE])
        else:
            Error("Bad Optimizer")
        
        num_thr = 99
        thresholds = [x for x in range(1,num_thr+1)]
        precision_metrics = [tf.keras.metrics.Precision(thresholds=1.0/(num_thr+1)*thr, name='p'+str(thr)) for thr in thresholds]
        recall_metrics = [tf.keras.metrics.Recall(thresholds=1.0/(num_thr+1)*thr, name='r'+str(thr)) for thr in thresholds]
        
        model.compile(loss = tf.losses.BinaryCrossentropy(), optimizer = optimizer, 
        metrics=['accuracy',
        precision_metrics, recall_metrics,
        tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name="tp"), tf.keras.metrics.TrueNegatives(name="tn"), 
        tf.keras.metrics.FalsePositives(name="fp"), tf.keras.metrics.FalseNegatives(name="fn")])
        
        if using_val_data:
            history = model.fit(
            train_features, train_labels, epochs=my_epochs, batch_size=hparams[HP_NUM_UNITS], 
            validation_data=(validation_features,validation_labels),
            class_weight=class_weights(),
            callbacks=[
                tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                hp.KerasCallback(logdir, hparams),  # log hparams
                ],
            verbose=0,
            )
        else:
            history = model.fit(train_features, train_labels, epochs=my_epochs, batch_size = hparams[HP_NUM_UNITS], 
            validation_split=my_validation_split,
            class_weight=class_weights(),
            callbacks=[
                tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                hp.KerasCallback(logdir, hparams),  # log hparams
                ],
            verbose=0,
            )
            
        f1,pr,re=[],[],[]
        mx=0
        opt_i = 0
        for i in range(1,num_thr+1):
            pr.append(history.history['val_p'+str(i)][-1])
            re.append(history.history['val_r'+str(i)][-1])
            f1.append(2.0*pr[-1]*re[-1]/(pr[-1]+re[-1]+10**(-7)))
        for i in range(num_thr):
            if f1[i]>mx:
                mx=f1[i]
                opt_i=i
        with open('thresholds.txt','a') as file:
            file.write(str(1.0/(1+num_thr)*(opt_i+1))+', '+str(pr[opt_i])+', '+str(re[opt_i])+', '+str(f1[opt_i])+'\n')

        tf.summary.scalar(TRAIN, model.count_params(), step=1)
        #print(hparams)
        hp.hparams(hparams)
        acc = history.history['accuracy'][-1]
        fn = history.history['fn'][-1]
        fp = history.history['fp'][-1]
        tn = history.history['tn'][-1]
        tp = history.history['tp'][-1]
        recall = history.history['recall'][-1]
        precision = history.history['precision'][-1]
        s = (tp+tn+fp+fn)
        tf.summary.scalar(ACC, acc, step=1)
        tf.summary.scalar(TP, 100*tp/s, step=1)
        tf.summary.scalar(TN, 100*tn/s, step=1)
        tf.summary.scalar(FP, 100*fp/s, step=1)
        tf.summary.scalar(FN, 100*fn/s, step=1)
        tf.summary.scalar(RECALL, recall, step=1)
        tf.summary.scalar(PRECISION, precision, step=1)
        tf.summary.scalar(F1, 2.0*(precision*recall)/(precision+recall+(0.1)**6), step=1)

        acc_val = history.history['val_accuracy'][-1]
        tp_val = history.history['val_tp'][-1]
        tn_val = history.history['val_tn'][-1]
        fp_val = history.history['val_fp'][-1]
        fn_val = history.history['val_fn'][-1]
        recall_val = history.history['val_recall'][-1]
        precision_val = history.history['val_precision'][-1]
        s = (tp_val+tn_val+fp_val+fn_val)
        tf.summary.scalar(ACC_VAL, acc_val, step=1)
        tf.summary.scalar(FN_VAL, 100*fn_val/s, step=1)
        tf.summary.scalar(FP_VAL, 100*fp_val/s, step=1)
        tf.summary.scalar(TN_VAL, 100*tn_val/s, step=1)
        tf.summary.scalar(TP_VAL, 100*tp_val/s, step=1)
        tf.summary.scalar(RECALL_VAL, recall_val, step=1)
        tf.summary.scalar(PRECISION_VAL, precision_val, step=1)
        tf.summary.scalar(F1_VAL, 2.0*(precision_val*recall_val)/(precision_val+recall_val+(0.1)**6), step=1)
        
        model.summary()

session_num = 0

#kfoldrun()
#sys.exit()

with open(threshold_file,'w') as file:
    file.write("Optimal threshold, Precision, Recall, F1-Score\n")

five_best_runs=[
    {
        HP_NUM_UNITS : 8192,
        HP_REGULARIZATION : 0.0033121,        
        HP_OPTIMIZER : 'adam',
        HP_NUM_LAYERS : 1,
        HP_LEARNING_RATE : 0.068801, 
        HP_REGULARIZE_FUNCTION : 'l2',
        HP_DROPOUT : 0.223833, 
        HP_LAYER_WIDTH : 1024
    },
    {
        HP_NUM_UNITS : 2048,
        HP_REGULARIZATION : 0.049355,        
        HP_OPTIMIZER : 'adam',
        HP_NUM_LAYERS : 2,
        HP_LEARNING_RATE : 0.001642, 
        HP_REGULARIZE_FUNCTION : 'l2',
        HP_DROPOUT : 0.451552, 
        HP_LAYER_WIDTH : 512
    },
    {
        HP_NUM_UNITS : 2048,
        HP_REGULARIZATION : 0.038835,        
        HP_OPTIMIZER : 'sgd',
        HP_NUM_LAYERS : 1,
        HP_LEARNING_RATE : 0.001555, 
        HP_REGULARIZE_FUNCTION : 'l1',
        HP_DROPOUT : 0.079757, 
        HP_LAYER_WIDTH : 1024
    },
    {
        HP_NUM_UNITS : 8192,
        HP_REGULARIZATION : 0.000052,        
        HP_OPTIMIZER : 'adam',
        HP_NUM_LAYERS : 0,
        HP_LEARNING_RATE : 0.021138, 
        HP_REGULARIZE_FUNCTION : 'l1',
        HP_DROPOUT : 0.140486, 
        HP_LAYER_WIDTH : 512
    },
    {
        HP_NUM_UNITS : 8192,
        HP_REGULARIZATION : 0.005826,        
        HP_OPTIMIZER : 'adam',
        HP_NUM_LAYERS : 0,
        HP_LEARNING_RATE : 0.029214, 
        HP_REGULARIZE_FUNCTION : 'l2',
        HP_DROPOUT : 0.776633, 
        HP_LAYER_WIDTH : 512
    }
]

for i in range(len(five_best_runs)):  
  
    hparams = five_best_runs[i]
  
    for i in range(num_trials):
    
        hparams[HP_NUM_TRIAL] = i
    
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(hparam_folder + '/' + run_name, hparams)
        session_num += 1