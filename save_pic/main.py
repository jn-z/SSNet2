import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot
import pdb
# read the log file
fp = open('log/cpc_NV10.log', 'r')
 

train_loss = []
train_accuracy = []
epoch_num = []
i=0
for ln in fp:
  # get epoch
  if 'Epoch:' in ln:
    epoch_num.append(float(ln.strip().split(':')[3][1:3]))
  # get train_loss
  if 'Loss:' in ln and ']' in ln:
    train_loss.append(float(ln.strip().split(':')[6][1:11]))#[0:5]
  # get train_accuracy
  if 'Accuracy:' in ln and ']' in ln:
    train_accuracy.append(float(ln.strip().split(':')[5][1:7]))#[0:5]
    i+=1
    
print(i)
#print(train_accuracy)
#print(train_loss)
#pdb.set_trace()
fp.close()
 
host = host_subplot(111)#1×1的图，第一张
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
par1 = host.twinx()#添加一个y进去
# set labels
host.set_ylabel("train loss")
par1.set_ylabel("train accuracy")
 
# plot curves
p1, = host.plot(train_loss, label="train_loss",linewidth=1)
p2, = par1.plot(train_accuracy, label="train accuracy",linewidth=1)
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
#host.legend(loc=5)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([0, i])
par1.set_ylim([0., 1.05])
plt.savefig('pic/cpc_NV10_trainall.png',dpi=1000)
plt.draw()
plt.show()

plt.close()