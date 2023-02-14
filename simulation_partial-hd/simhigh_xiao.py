# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:29:31 2021

@author: public
"""
import os
current_path = os.path.abspath(__file__)
current_path = os.path.dirname(os.path.abspath(__file__))
# current_path = os.getcwd()
os.chdir(current_path)

import os
import time 
import numpy as np
import numpy.linalg as la
from simulation_fun.simhigh.generate_data import *
from simulation_fun.simhigh.readresult import *
from simulation_fun.simhigh.simultion import *
# import sys
# sys.path.append("..")
# sys.path.append("..")
from method_fun.high.tuning import *
from method_fun.high.semi_fun import *

indexpath0 = current_path+'\\results'
if not os.path.exists(indexpath0):
    os.mkdir(indexpath0)
indexpath0 = current_path+'\\results\\simhigh'
if not os.path.exists(indexpath0):
    os.mkdir(indexpath0)
os.chdir(indexpath0)

import pickle
"""
#1. simulation
"""  
ratio = 2/3
par_1_default = [100,1000,0.1,6,20]
par_2_default = [1,-0.3,'EPA','Gaussian',1e-5,1e-5,1e-5]
ntimes = 500
errornum = 5

import pickle
import random
for i in range(errornum):
    list_file = open('errror'+str(i)+'result'+'-all.pickle','rb')
    resulterri = pickle.load(list_file)
    betaall,parmus,nonpamus,allmus,ldssre,hallre = resulterri
    for j in range(ntimes):
        random.seed(j) 
        np.random.seed(j)
        print('errror'+str(i)+'-result'+str(j)+'-th-'+str(np.round(ratio,2))+'ld.pickle')
        ldss, hall = ldssre[j,:].copy(),hallre[j,:].copy()
        resulterri = predsim_diffld(ldss,hall,i,par_1_default,par_2_default,ratio)
        #存原始结果
        list_file = open('errror'+str(i)+'-result'+str(j)+'-th-'+str(np.round(ratio,2))+'ld.pickle','wb')
        pickle.dump(resulterri,list_file)
        print(time.strftime('%H.%M.%S',time.localtime(time.time())))
        list_file.close()
"""
#2. summary results
""" 

#2.1 summary initial results and save
n,p,ldmin,ldmax,nld = par_1_default
for i in range(errornum):
    betaall = np.zeros((ntimes,p,6))
    allmus = np.zeros((ntimes,int(n/2),6))
    parmus = np.zeros((ntimes,int(n/2),6))
    nonpamus = np.zeros((ntimes,int(n/2),6))
    ldssre2 = np.zeros((ntimes,6))
    hallre2 = np.zeros((ntimes,6))
    for j in range(ntimes):
        list_file = open('errror'+str(i)+'-result'+str(j)+'-th-'+str(np.round(ratio,2))+'ld.pickle','rb')
        resulterri = pickle.load(list_file)
        betaallj,parmusj,nonpamusj,allmusj,ldssrej,hallrej,dataallj = resulterri
        betaall[j],parmus[j],nonpamus[j],allmus[j],ldssre2[j],hallre2[j] = betaallj,parmusj,nonpamusj,allmusj,ldssrej,hallrej
    resulterriall = [betaall,parmus,nonpamus,allmus,ldssre2,hallre2]  
    list_file = open('errror'+str(i)+'result'+'-all'+str(np.round(ratio,2))+'ld.pickle','wb')
    pickle.dump(resulterriall,list_file)
    list_file.close()


#2.2 change the format and save

#read
resultall = []
import pickle
for i in range(errornum):
    list_file = open('errror'+str(i)+'result'+'-all'+str(np.round(ratio,2))+'ld.pickle','rb')
    resulterri = pickle.load(list_file)
    resultall.append(resulterri)

#change
finalresultcv=np.zeros((5,7*6*2))
for i in range(errornum):
    resulterri = resultall[i]
    finalresultcv[i,:] = calestimate(ntimes,resulterri)
    print('Final',i) 
finalresultcv2=np.zeros((5,7*6*2))
for i in range(errornum):
    resulterri = resultall[i]
    finalresultcv2[i,:] = calestimatemap(ntimes,resulterri)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))

#save
import pickle
list1 = [resultall,finalresultcv,finalresultcv2]
list_file = open('simhigh'+str(np.round(ratio,2))+'ld.pickle','wb')
pickle.dump(list1,list_file)
list_file.close()
  


"""
3. 存latex
"""

#read
list_file = open('simhigh'+str(np.round(ratio,2))+'ld.pickle','rb')
list2 = pickle.load(list_file)
resultall,finalresultcv,finalresultcv2 = list2

#change&save
'''table1all'''
result=finalresultcv.copy()
result=np.array(result).reshape(5,7*6*2)
result=np.round(result,3)
data3=open('table5all'+str(np.round(ratio,2))+'ld.txt','w+') 
for j in range(errornum):
    temp=result[j,:]
    print(j)
    print('$'+'C'+str(j+1)+'$'+'&'+'PPLSE'+'&'+str(temp[0+12])+'('+str(temp[6+12])+')'+'&'+str(temp[3+12])+'('+str(temp[9+12])+')'+'&'+str(temp[0+36])+'('+str(temp[6+36])+')'+'&'+str(temp[3+36])+'('+str(temp[9+36])+')'+'&'+str(temp[0])+'&'+str(temp[3])+'&'+str(temp[72])+'&'+str(temp[3+72])+'\\'+'\\'+'\n',file=data3)
    print('&'+'PSEMI*'+'&'+ str(temp[1+12])+'('+str(temp[7+12])+')'+'&'+str(temp[4+12])+'('+str(temp[10+12])+')'+'&'+ str(temp[1+36])+'('+str(temp[7+36])+')'+'&'+str(temp[4+36])+'('+str(temp[10+36])+')'+'&'+str(temp[1])+'&'+str(temp[4])+'&'+str(temp[1+72])+'&'+str(temp[4+72])+'\\'+'\\'+'\n',file=data3)
    print('&'+'PSEMI'+'&'+ str(temp[2+12])+'('+str(temp[8+12])+')'+'&'+str(temp[5+12])+'('+str(temp[11+12])+')'+'&'+ str(temp[2+36])+'('+str(temp[8+36])+')'+'&'+str(temp[5+36])+'('+str(temp[11+36])+')'+'&'+str(temp[2])+'&'+str(temp[5])+'&'+str(temp[2+72])+'&'+str(temp[5+72])+'\\'+'\\'+'\n',file=data3)
data3.close()
