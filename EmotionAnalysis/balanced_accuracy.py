# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:26:22 2020

@author: Allen
"""
import numpy as np

result_file=u'datafiles/result.txt'
result_list=[]
label_list =[]

with open('datafiles/label_example.txt') as f:
    for line in f.readlines():
        line=line.strip()
        label_list.append(line)
        
with open(result_file) as f:
    for line in f.readlines():
        line=line.strip()
        result_list.append(line)

llist=np.array(label_list)
rlist=np.array(result_list)

if(len(rlist[llist=='p'])==0):
    pacc=0
else:
    pacc=sum(rlist[llist=='p']=='p')*1.0/sum(llist=='p')
if len(rlist[llist=='n'])==0:
    nacc=0
else:
    nacc=sum(rlist[llist=='n']=='n')*1.0/sum(llist=='n')

fname=result_file.split('.')[0]

print('%s: pacc:%s, nacc:%s, bacc:%s'%(fname,pacc,nacc ,(pacc+nacc)/2))

