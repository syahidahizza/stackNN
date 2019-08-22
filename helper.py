import os
import pickle
import numpy as np

from os import listdir
from os.path import isfile, join

from scipy.stats import entropy
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from matplotlib import mlab
import matplotlib.pyplot as plt

def load_probability(merge_net_name, data_dir='prediction_imagenet/') :
    x_val = pickle.load(open(data_dir+'val_x_data_'+merge_net_name+'.dmp', 'rb'))
    y_val = pickle.load(open(data_dir+'val_y_data_'+merge_net_name+'.dmp', 'rb'))
    model = joblib.load('prediction_imagenet/lgb_'+merge_net_name+'.pkl')
    y_pred = model.predict_proba(x_val)
    return y_pred[:, 1], y_val

def load_logit_acc(net_name, pred_name, n_val=50000, n_feature = 1001, data_dir='prediction_imagenet/') :
    dir_val1 = data_dir+net_name+'/validation/'
    val_file1 = [f for f in listdir(dir_val1) if (isfile(join(dir_val1, f)) and ('filename_mapping' not in f))]
    val_data_helper = {}

    for x in val_file1:
        curr_data = pickle.load(open(dir_val1+x,'rb'))
        for image_file_name in curr_data.keys():
            data_point = curr_data[image_file_name]
            if image_file_name not in val_data_helper.keys():
                val_data_helper[image_file_name] = {}
            val_data_helper[image_file_name][pred_name] = 1 if data_point[pred_name] == data_point['label'] else 0
            val_data_helper[image_file_name]['last_layer'] = data_point['last_layer_'+net_name]
        
    offset=0
    logits1 = np.zeros((n_val,n_feature))
    acc_val_net1 = np.zeros((n_val))
    keys = pickle.load(open(data_dir+'sequence.dmp', 'rb'))
    for x in keys:
        logits1[offset] = val_data_helper[x]['last_layer']
        acc_val_net1[offset] = val_data_helper[x][pred_name]
        offset+=1

    return logits1, acc_val_net1

def show(all_thresholds, all_accuracy, all_speedup, title) :
    thresholds = all_thresholds.reshape(100, -1)
    thresholds = np.hstack((thresholds[:, 0], thresholds[-1, -1]))
    accuracy = all_accuracy.reshape(100, -1)
    accuracy = np.hstack((accuracy[:, 0], accuracy[-1, -1]))
    speedup = all_speedup.reshape(100, -1)
    speedup = np.hstack((speedup[:, 0], speedup[-1, -1]))
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_ylabel('Accuracy Loss', fontsize=20)  
    ax1.plot(thresholds, accuracy, c='lime', label='accuracy', linewidth=3)
    ax1.plot(np.nan, np.nan, c='orange', label = 'speedup', linewidth=3)

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Speedup', fontsize=20)  
    ax2.plot(thresholds, speedup, c='orange', label='speedup', linewidth=3)
    ax2.tick_params(axis='y')
    ax2.tick_params(labelsize=15)

    ax1.set_xlabel('Decider Threshold', fontsize=20)  
    ax1.legend(loc=4, fontsize = 15)
    ax1.tick_params(labelsize=15)
    
    ax1.set_title(title, fontsize=25)
    fig.tight_layout() 
    ax1.grid(True)
    plt.show()
    
def show_auc(y_val, y_pred, entropy_val, auc_entropy, auc_boosting):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlabel('False Positive Rate', fontsize=20)  
    ax1.set_ylabel('True Positive Rate', fontsize=20)  

    fpr, tpr, threshold = roc_curve(y_val, entropy_val)
    ax1.plot(fpr, tpr, c='lime', label='ROC entropy (area = %0.2f)' % auc_entropy, linewidth=3)

    fpr, tpr, threshold = roc_curve(y_val, y_pred)
    ax1.plot(fpr, tpr, c='orange', label='ROC Boosting (area = %0.2f)' % auc_boosting, linewidth=3)

    ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', linewidth=3)
    ax1.legend(loc=4, fontsize = 15)
    ax1.tick_params(labelsize=15)

    fig.tight_layout() 
    ax1.set_title('ROC Entropy vs ROC Boosting', fontsize=25)
    ax1.grid()
    plt.show()
    
def calculate_speedup(all_thresholds, accuracy1, accuracy2) :
    n_val = len(accuracy1)
    mean_accuracy2 = np.mean(accuracy2)
    passed_rate = []
    accuracy = []

    idx_01 = (accuracy1 == 0) & (accuracy2 == 1)
    val_01 = []
    idx_10 = (accuracy1 == 1) & (accuracy2 == 0)
    val_10 = []

    for t in all_thresholds :
        acc1 = np.sum(accuracy1[all_thresholds<=t])
        acc2 = np.sum(accuracy2[all_thresholds>t])
        accuracy.append(mean_accuracy2-np.sum([acc1, acc2])/n_val)
        
        curr_01 = np.sum(idx_01[all_thresholds>t])/n_val
        val_01.append(curr_01)

        curr_10 = np.sum(idx_10[all_thresholds>t])/n_val
        val_10.append(curr_10)

        small_count = np.mean(all_thresholds<=t)
        large_count = 1 - small_count
        passed = np.mean(all_thresholds>t)
        passed_rate.append(passed)

    sorted_index = np.argsort(all_thresholds)
    all_thresholds = all_thresholds[sorted_index]
    passed_rate = np.array(passed_rate)[sorted_index]
    accuracy = np.array(accuracy)[sorted_index]
    
    val_01 = np.array(val_01)[sorted_index]
    val_10 = np.array(val_10)[sorted_index]
    
    return all_thresholds, accuracy, passed_rate, val_01, val_10