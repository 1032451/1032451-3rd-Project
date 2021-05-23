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
import os

#------------------------------------------------------------------------------
my_seed = 1
np.random.seed(my_seed)
rng = np.random.default_rng(my_seed)

def load_features(file_name):
    # dictionary to load features into
    features = {"CM":[],"CNH":[],"HOG":[],"LBP":[]}
    if not os.path.isfile(file_name):
        return features
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
    if not os.path.isfile(file_name):
        return []
    with open(file_name) as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i].split(" ")
        if(len(content[i])<2):
            return content[:i]
        content[i] = [int(content[i][0]),int(content[i][1])]
    #print(content) 
    return content

#------------------------------------------------------------------------------

#video_features = load_features('Hollywood-dev/features/SavingPrivateRyan_visual.mat')
#annotations = load_annotations('Hollywood-dev/annotations/SavingPrivateRyan_blood.txt')

names_high_level_violence_concepts = ['Armageddon','BillyElliot','DeadPoetsSociety','Eragon','FightClub','HarryPotterTheOrderOfThePhoenix','IAmLegend',
'IndependanceDay','Leon','MidnightExpress','PiratesOfTheCarribeanTheCurseOfTheBlackPearl','ReservoirDogs','SavingPrivateRyan','TheBourneIdentity',
'TheSixthSense','TheWickerMan','TheWizardOfOz']

names_train = ['Armageddon','BillyElliot','DeadPoetsSociety','Eragon','FantasticFour','Fargo','FightClub','ForrestGump',
'HarryPotterTheOrderOfThePhoenix','IAmLegend','IndependanceDay','LegallyBlonde','Leon','MidnightExpress','PiratesOfTheCarribeanTheCurseOfTheBlackPearl',
'PulpFiction','ReservoirDogs','SavingPrivateRyan','TheBourneIdentity','TheGodFather','ThePianist','TheSixthSense','TheWickerMan','TheWizardOfOz']

names_test = ['8Mile','Braveheart','Desperado','GhostintheShell','Jumanji','Terminator2','VforVendetta']

#movies_train = [(['Hollywood-dev/annotations/'+n+'_blood.txt','Hollywood-dev/annotations/'+n+'_fire.txt','Hollywood-dev/annotations/'+n+'_gore.txt','Hollywood-dev/annotations/'+n+'_violence.txt'],'Hollywood-dev/features/'+n+'_visual.mat') for n in names_train]
movies_train = [(['Hollywood-dev/annotations/'+n+'_violence.txt'],'Hollywood-dev/features/'+n+'_visual.mat') for n in names_train]
movies_test = [(['Hollywood-test/annotations/'+n+'_violence.txt'],'Hollywood-test/features/'+n+'_visual.mat') for n in names_test]

concepts = ['blood','carchase','coldarms','fights','fire','firearms','gore']

no_frames = sum(len(load_features('Hollywood-dev/features/'+n+'_visual.mat')['CM']) for n in names_train)
print("Number of frames:"+str(no_frames))
s = sum([len(load_annotations(a[0])) for (a,v) in movies_train])
print("Number of violent scenes:"+str(s))
#sys.exit()

def violent_percentages():
    percentage,v,nv = {},{},{}
    for c in concepts:
        print(c)
        percentage[c]=0.0
        v[c] = 0
        nv[c] = 0
    for m in names_high_level_violence_concepts:
        video_features = load_features('Hollywood-dev/features/'+m+'_visual.mat')
        no_frames = len(video_features['CM'])
        
        for c in concepts:
        
            violent=[0]*(no_frames)

            annotations = load_annotations('Hollywood-dev/annotations/'+m+'_'+c+'.txt')
            for an in annotations:
                for i in range(an[0],an[1]+1):
                    violent[i]=1
        
            nv[c]+=no_frames
            v[c]+=len([x for x in violent if x])
        
    for c in concepts:
        percentage[c] = 100.0*v[c]/nv[c]
        
    return percentage
        
            

def make_dataset(movies,file_name):

    with open(file_name, "w") as file:
        file.write(",".join([str(x) for x in range(81+99+81+144)])+",V\n")

    for (a,v) in movies:

        video_features = load_features(v)
        no_frames = len(video_features["CM"])

        violent=[0]*(no_frames)

        for trait in a:
            print(trait)
            annotations = load_annotations(trait)
            print(len(annotations))
            for an in annotations:
                for i in range(an[0],an[1]+1):
                    violent[i]=1

        if no_frames>0:
            violent_percentage = 1.0*len([a for x in violent if x])/no_frames
            print(violent_percentage*100)
        
        with open(file_name, "a") as file:
            for i in range(0,no_frames,24):
                s = str(np.char.join(',',map(str,video_features["CM"][i])))+","+str(np.char.join(',',map(str,video_features["CNH"][i])))+","+str(np.char.join(',',map(str,video_features["HOG"][i])))+","+str(np.char.join(',',map(str,video_features["LBP"][i])))
                file.write(s+","+str(violent[i])+"\n")


print(violent_percentages())

make_dataset(movies_train,"dataset_train_small.csv")
make_dataset(movies_test,"dataset_test_small.csv")