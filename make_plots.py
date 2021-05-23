import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os

folder = 'metrics/'

#run-run-0_validation-tag-epoch_accuracy.csv

def plot_tag(tag,val=True):

    tr_or_val = 'train'
    if val:
        tr_or_val = 'validation'
    names = ['run-run-'+str(i)+'_'+tr_or_val+'-tag-'+tag+'.csv' for i in range(0,10)]

    with open(folder+'merged.csv','w') as file:
        file.write("N,Wall Time,Step,Value\n")

    i=0
    for name in names:
        with open(folder+'merged.csv','a') as w:
            with open(folder+name,'r') as r:
                for line in r.readlines()[1:]:
                    w.write(str(i)+","+line)
        i+=1
    
    data = pd.read_csv(folder+"merged.csv")

    plot = sns.lineplot(data=data,x="Step",y="Value")

    #plt.show()

for tag in ['epoch_accuracy','epoch_loss','epoch_precision','epoch_recall']:    
    for val in [True,False]:
        plot_tag(tag,val)
        plt.show()
    
#plt.show()