# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:15:39 2021

@author: public

说明：当n越来越大的时候semi就越慢，且该方法中semi和semi*以及lse的最优h都差不多，所以semi*和semi在计算预测表的时候用的是不同h，方差表和估计表的时候用的是同一个h
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
from simulation_fun.simlow.generate_data import *
from simulation_fun.simlow.readresult import *
from simulation_fun.simlow.simultion import *
# import sys
# sys.path.append("..")
# sys.path.append("..")
from method_fun.low.semi_func import *

indexpath0 = current_path+'\\results'
if not os.path.exists(indexpath0):
    os.mkdir(indexpath0)
indexpath0 = current_path+'\\results\\simlow'
if not os.path.exists(indexpath0):
    os.mkdir(indexpath0)
os.chdir(indexpath0)

import pickle



import time

import os
current_path = os.path.abspath(__file__)
indexpath0 = os.path.dirname(os.path.abspath(__file__))
indexpath= indexpath0+'\\simlow-0330'
if not os.path.exists(indexpath):
    os.mkdir(indexpath)
os.chdir(indexpath)

'''------------------------------------------------------------------------------------------------------'''
'''4. main---simulation-table1
---4.1 table6-prederror
  ---simulation
    ---predsim
  ---write result
    ---calestimate
---4.2 table1-mse/mad:
  ---simulation
    ---KDRESIMsemi
    ---注：用上一段（predsim）和下一段（KDRESIMsemi）都行，这里用的是下一段
  ---write result
    ---writemse：存mse结果
    ---writemad：存mad结果
---4.3 table2-SD
  ---simulation
    ---sdestlse/KDRESIMlse: 估计lse方差的函数/simulation函数
    ---sdestsemi/KDRESIMsemi: 估计semi方差的函数/simulation函数
  ---write result
    ---KDRESIM: 存SEMI*-SD结果
    ---KDRESIM2：存SEMI-SD结果
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


'''------------------------------------------------------------------------------------------------------'''
'''5. main---simulation-table-prederror
---5.1 计算
---5.2 存结果
---5.3 存latex
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


"""
1. 计算 n = 30
"""

import random
   
n=30
ntimes=100
resultall30 = []
for i in range(5):
    random.seed(1) 
    np.random.seed(1)
    resulterri = predsim(ntimes,i,n,alpha=-0.30,typek ='EPA',typek2 ='Gaussian')
    resultall30.append(resulterri)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    


"""
2. 存结果
"""      

finalresultcv=np.zeros((5,7*3*2))
for i in range(5):
    resulterri = resultall30[i]
    finalresultcv[i,:] = calestimate(ntimes,resulterri,n)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))

finalresultcv2=np.zeros((5,7*3*2))
for i in range(5):
    resulterri = resultall30[i]
    finalresultcv2[i,:] = calestimatemap(ntimes,resulterri,n)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))



#存原始结果
import pickle
list1 = [resultall30,finalresultcv,finalresultcv2]
list_file = open('re30.pickle','wb')
pickle.dump(list1,list_file)
list_file.close()

"""
#读所有结果
"""  
import pickle
n =30  
p = 3
list_file = open('re30.pickle','rb')
list2 = pickle.load(list_file)
resultall,finalresultcv,finalresultcv2 = list2
#further 分析
betaall = np.zeros((5,ntimes,p,3))
parmus = np.zeros((5,ntimes,int(n/2),3))
nonpamus = np.zeros((5,ntimes,int(n/2),3))
allmus= np.zeros((5,ntimes,int(n/2),3))
ldssre = np.zeros((5,ntimes,3))
dataall = np.zeros((5,ntimes,n,p+2))
hallre = np.zeros((5,ntimes,3))
data = []
for i in range(5):
    betaall[i,:],parmus[i,:],nonpamus[i,:],allmus[i,:],hallre[i,:],dataallre = resultall[i]
    data.append(dataallre)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
for i in range(5):
    for j in range(ntimes):
        dataall[i,j,:,:-2] = data[i][j][0]
        dataall[i,j,:,-2] = data[i][j][1].ravel()
        dataall[i,j,:,-1] = data[i][j][2].ravel()

"""
3. 存latex
"""

'''table12mean'''
result=finalresultcv.copy()
result=np.array(result).reshape(5,7*3*2)
result=np.round(result,3)
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
data4=open(indexpath+ '\\table6mean30.txt','w+') 
for j in range(5):
    temp=result[j,:]
    print(j)
    print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'&'+str(temp[0+30])+'&'+ str(temp[0+31])+'&'+str(temp[0+32])+'\\'+'\\'+'\n',file=data4)
    # print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+
    #       str(temp[0+25])+'&'+str(temp[0+26])+'\\'+'\\'+'\n',file=data4)
data4.close()
 
  
'''table12-map'''
result2=finalresultcv2.copy()
result2=np.array(result2).reshape(5,7*3*2)
result2=np.round(result2,3)
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
data4=open(indexpath+ '\\table6mape30.txt','w+') 
for j in range(5):
    temp=result2[j,:]
    print(j)
    print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'&'+str(temp[0+30])+'&'+ str(temp[0+31])+'&'+str(temp[0+32])+'\\'+'\\'+'\n',file=data4)
    # print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'
    #       +str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'\\'+'\\'+'\n',file=data4)
data4.close()


"""
1. 计算 n=100
"""

n=100
ntimes=100
resultall100 = []
for i in range(5):
    random.seed(1) 
    np.random.seed(1)
    resulterri = predsim(ntimes,i,n,alpha=-0.30,typek ='EPA',typek2 ='Gaussian')
    resultall100.append(resulterri)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    
    
        
finalresultcv=np.zeros((5,7*3*2))
for i in range(5):
    resulterri = resultall100[i]
    finalresultcv[i,:] = calestimate(ntimes,resulterri,n)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))

finalresultcv2=np.zeros((5,7*3*2))
for i in range(5):
    resulterri = resultall100[i]
    finalresultcv2[i,:] = calestimatemap(ntimes,resulterri,n)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))



#存原始结果
import pickle
list1 = [resultall100,finalresultcv,finalresultcv2]
list_file = open('re100.pickle','wb')
pickle.dump(list1,list_file)
list_file.close()
  
"""
#读所有结果
"""  
n=100  
p = 3
list_file = open('re100.pickle','rb')
list2 = pickle.load(list_file)
resultall100,finalresultcv,finalresultcv2 = list2
#further 分析
betaall = np.zeros((5,ntimes,p,3))
parmus = np.zeros((5,ntimes,int(n/2),3))
nonpamus = np.zeros((5,ntimes,int(n/2),3))
allmus= np.zeros((5,ntimes,int(n/2),3))
ldssre = np.zeros((5,ntimes,3))
dataall = np.zeros((5,ntimes,n,p+2))
hallre = np.zeros((5,ntimes,3))
data = []
for i in range(5):
    betaall[i,:],parmus[i,:],nonpamus[i,:],allmus[i,:],hallre[i,:],dataallre = resultall100[i]
    data.append(dataallre)
    print('Final',i) 
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
for i in range(5):
    for j in range(ntimes):
        dataall[i,j,:,:-2] = data[i][j][0]
        dataall[i,j,:,-2] = data[i][j][1].ravel()
        dataall[i,j,:,-1] = data[i][j][2].ravel()

"""
3. 存latex
"""

result=finalresultcv.copy()
result=np.array(result).reshape(5,7*3*2)
result=np.round(result,3)
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
data4=open(indexpath+ '\\table6mean100.txt','w+') 
for j in range(5):
    temp=result[j,:]
    print(j)
    print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'&'+str(temp[0+30])+'&'+ str(temp[0+31])+'&'+str(temp[0+32])+'\\'+'\\'+'\n',file=data4)
    # print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+
    #       str(temp[0+25])+'&'+str(temp[0+26])+'\\'+'\\'+'\n',file=data4)
data4.close()
 
  
'''table12-map'''
result2=finalresultcv2.copy()
result2=np.array(result2).reshape(5,7*3*2)
result2=np.round(result2,3)
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
data4=open(indexpath+ '\\table6mape100.txt','w+') 
for j in range(5):
    temp=result2[j,:]
    print(j)
    print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'+str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'&'+str(temp[0+30])+'&'+ str(temp[0+31])+'&'+str(temp[0+32])+'\\'+'\\'+'\n',file=data4)
    # print('$'+'C'+str(j+1)+'$'+'&'+str(temp[0+12])+'&'+ str(temp[0+13])+'&'+str(temp[0+14])+'&'
    #       +str(temp[0+24])+'&'+ str(temp[0+25])+'&'+str(temp[0+26])+'\\'+'\\'+'\n',file=data4)
data4.close()

'''------------------------------------------------------------------------------------------------------'''
'''6. main---simulation-table-mse
---6.1 计算
---6.2 存结果
---6.3 存latex
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''

"""
1. 计算
""" 

import random
reall300 = []
for i in range(5):
    random.seed(1) 
    np.random.seed(1)
    b = KDRESIMsemi(200,i,300,alpha = -0.3,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5)
    #b = reall100[i+5]
    reall300.append(b)
    # bb = np.float64(np.array(b)[:-1:,:,:])
    # re30.append(np.vstack((np.std(np.array(bb)[-3:,:200,:],1),np.mean(np.array(bb)[:-3,:200,:],1))))


reall100 = []
for i in range(5):
    random.seed(1) 
    np.random.seed(1)  
    b = KDRESIMsemi(200,i,100,alpha = -0.3,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5)
    reall100.append(b)
    # bb = np.float64(np.array(b)[:-1:,:,:])
    # re30.append(np.vstack((np.std(np.array(bb)[-3:,:200,:],1),np.mean(np.array(bb)[:-3,:200,:],1))))

reall30 = []
for i in range(5):
    random.seed(1) 
    np.random.seed(1)       
    b = KDRESIMsemi(200,i,30,alpha = -0.3,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5)
    reall30.append(b)



"""
2. 存结果
""" 

import pickle
list1 = [reall300]
list_file = open('reall300.pickle','wb')
pickle.dump(list1,list_file)
list_file.close()


import pickle
list1 = [reall30,reall100,reall300]
list_file = open('reall.pickle','wb')
pickle.dump(list1,list_file)
list_file.close()




"""
3. 存tex
"""  

import pickle
list_file = open('reall300.pickle','rb')
reall300 = pickle.load(list_file)
list_file.close()


import pickle
list_file = open('reall.pickle','rb')
reall30,reall100,reall300 = pickle.load(list_file)
list_file.close()




re30 = []
for i in range(5):
    b = reall30[i]
    bb = np.float64(np.array(b)[-4:-1,:ntimes,:])
    re30.append(writemse(ntimes,i,bb))
    
re100 = []
for i in range(5):
    b = reall100[i]
    bb = np.float64(np.array(b)[-4:-1,:ntimes,:])
    re100.append(writemse(ntimes,i,bb))   


finalresultmse = np.vstack((np.array(re30),np.array(re100)))

finalresultmse3=np.array(finalresultmse).reshape(10,18)
tt=[30,30,30,30,30,100,100,100,100,100]
aa=finalresultmse3.copy()
aa=np.round(aa,3)
data=open(indexpath+ '\\table1-mse.txt','w+') 
for i in range(10):
#    print('$'+'C'+str(i%5+1)+'-'+'{'+str(tt[i])+'}$'+'&'+str(aa[i,0+0])+'('+str(aa[i,0+3])+')'+'&'+str(aa[i,0+6])+'('+str(aa[i,0+9])+')'+'&'+str(aa[i,0+12])+'('+str(aa[i,0+15])+')'+'&'+str(aa[i,1+0])+'('+str(aa[i,1+3])+')'+'&'+str(aa[i,1+6])+'('+str(aa[i,1+9])+')'+'&'+str(aa[i,1+12])+'('+str(aa[i,1+15])+')'+'&'+str(aa[i,2+0])+'('+str(aa[i,2+3])+')'+'&'+str(aa[i,2+6])+'('+str(aa[i,2+9])+')'+'&'+str(aa[i,2+12])+'('+str(aa[i,2+15])+')'+'\\'+'\\',file=data)
    print('&'+'$'+'C'+str(i%5+1)+'-'+'{'+str(tt[i])+'}$'+'&'+str(aa[i,0+0])+'&'+str(aa[i,0+6])+'&'+str(aa[i,0+12])+'&'+str(aa[i,1+0])+'&'+str(aa[i,1+6])+'&'+str(aa[i,1+12])+'&'+str(aa[i,2+0])+'&'+str(aa[i,2+6])+'&'+str(aa[i,2+12])+'\\'+'\\',file=data)
data.close()


re30 = []
for i in range(5):
    b = reall30[i]
    bb = np.float64(np.array(b)[-4:-1,:ntimes,:])
    re30.append(writemad(ntimes,i,bb))
    
re100 = []
for i in range(5):
    b = reall100[i]
    bb = np.float64(np.array(b)[-4:-1,:ntimes,:])
    re100.append(writemad(ntimes,i,bb))   


finalresultmad = np.vstack((np.array(re30),np.array(re100)))

finalresultmse3f=np.array(finalresultmad).reshape(10,18)
tt=[30,30,30,30,30,100,100,100,100,100]
aa=finalresultmse3f.copy()
aa=np.round(aa,3)
data=open(indexpath+ '\\table1-mad.txt','w+') 
for i in range(10):
    print('$'+'C'+str(i%5+1)+'-'+'{'+str(tt[i])+'}$'+'&'+str(aa[i,0+0])+'&'+str(aa[i,0+6])+'&'+str(aa[i,0+12])+'&'+str(aa[i,1+0])+'&'+str(aa[i,1+6])+'&'+str(aa[i,1+12])+'&'+str(aa[i,2+0])+'&'+str(aa[i,2+6])+'&'+str(aa[i,2+12])+'\\'+'\\',file=data)
data.close()


'''------------------------------------------------------------------------------------------------------'''
'''6. main---simulation-table2-SD
---6.1 计算
---6.2 存结果
---6.3 存latex
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''



# re30 = []
# for i in range(5):
#     b = reall30[i]
#     bb = np.float64(np.array(b)[:-1:,:,:])
#     re30.append(np.vstack((np.nanstd(np.array(bb)[-3:,:100,:],1),np.nanmean(np.array(bb)[:-3,:100,:],1))))
    
# re100 = []
# for i in range(5):
#     b = reall100[i]
#     bb = np.float64(np.array(b)[:-1:,:,:])
#     re100.append(np.vstack((np.nanstd(np.array(bb)[-3:,:100,:],1),np.nanmean(np.array(bb)[:-3,:100,:],1))))
    

re300 = []
for i in range(5):
    b = reall300[i]
    bb = np.float64(np.array(b)[:-1:,:,:])
    re300.append(np.vstack((np.nanstd(np.array(bb)[-3:,:200,:],1),np.nanmean(np.array(bb)[:-3,:200,:],1)-3*np.nanstd(np.array(bb)[:-3,:200,:],1))))



"""
存latex for SEMI*
"""


import time
#模拟200次，n=300
j=300

finalresultsd1=[]
for i in range(5):
    b = reall300[i]
    bb = np.float64(np.array(b)[:-1:,:,:])
    finalresultsd1.append(KDRESIM(ntimes,i,bb))
    print(i)
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
finalresultsd1=np.array(finalresultsd1).reshape(5,24)
aa=finalresultsd1.copy()



#6;12;18
#6:tmp:local eff(profile lse的真实值);
#12:tmp2：eff的近似值；
#18:tmp3 eff的真实值？；
'''sd3>sd>sd2: 18>6>12'''
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
test=np.array(aa).copy()
test=np.round(test,3)
data=open(indexpath+'\\LEFF-SD-final.txt','w+') 
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,6+0])+ '&' +'('+str(np.round(test[i,6+0]-3*test[i,6+0+3],3))+','+str(np.round(test[i,6+0]+3*test[i,6+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,6+1])+ '&' +'('+str(np.round(test[i,6+1]-3*test[i,6+1+3],3))+','+str(np.round(test[i,6+1]+3*test[i,6+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,6+2])+ '&' +'('+str(np.round(test[i,6+2]-3*test[i,6+2+3],3))+','+str(np.round(test[i,6+2]+3*test[i,6+2+3],3))+')'+
          '\\'+'\\',file=data)
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,12+0])+ '&' +'('+str(np.round(test[i,12+0]-3*test[i,12+0+3],3))+','+str(np.round(test[i,12+0]+3*test[i,12+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,12+1])+ '&' +'('+str(np.round(test[i,12+1]-3*test[i,12+1+3],3))+','+str(np.round(test[i,12+1]+3*test[i,12+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,12+2])+ '&' +'('+str(np.round(test[i,12+2]-3*test[i,12+2+3],3))+','+str(np.round(test[i,12+2]+3*test[i,12+2+3],3))+')'+
          '\\'+'\\',file=data)
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,18+0])+ '&' +'('+str(np.round(test[i,18+0]-3*test[i,18+0+3],3))+','+str(np.round(test[i,18+0]+3*test[i,18+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,18+1])+ '&' +'('+str(np.round(test[i,18+1]-3*test[i,18+1+3],3))+','+str(np.round(test[i,18+1]+3*test[i,18+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,18+2])+ '&' +'('+str(np.round(test[i,18+2]-3*test[i,18+2+3],3))+','+str(np.round(test[i,18+2]+3*test[i,18+2+3],3))+')'+
          '\\'+'\\',file=data)
data.close() 



"""
存latex for SEMI
"""


import time
#模拟200次，n=300
j=300

finalresultsd1=[]
for i in range(5):
    b = reall300[i]
    bb = np.float64(np.array(b)[:-1:,:,:])
    finalresultsd1.append(KDRESIM2(ntimes,i,bb))
    print(i)
    print(time.strftime('%H.%M.%S',time.localtime(time.time())))
finalresultsd1=np.array(finalresultsd1).reshape(5,24)
aa=finalresultsd1.copy()


#6;12;18
#6:tmp 独立的logeta`^2;
#12:tmp2独立的-logeta''；
#18:tmp3 总的logeta`^2
tt=['_{normal}$','_{skew}$','_{heavy-tail}$','_{outlier}$','_{multi-model}$']
test=np.array(aa).copy()
test=np.round(test,3)
data=open(indexpath+'\\EFF-SD-EFF.txt','w+') 
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,6+0])+ '&' +'('+str(np.round(test[i,6+0]-3*test[i,6+0+3],3))+','+str(np.round(test[i,6+0]+3*test[i,6+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,6+1])+ '&' +'('+str(np.round(test[i,6+1]-3*test[i,6+1+3],3))+','+str(np.round(test[i,6+1]+3*test[i,6+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,6+2])+ '&' +'('+str(np.round(test[i,6+2]-3*test[i,6+2+3],3))+','+str(np.round(test[i,6+2]+3*test[i,6+2+3],3))+')'+
          '\\'+'\\',file=data)
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,12+0])+ '&' +'('+str(np.round(test[i,12+0]-3*test[i,12+0+3],3))+','+str(np.round(test[i,12+0]+3*test[i,12+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,12+1])+ '&' +'('+str(np.round(test[i,12+1]-3*test[i,12+1+3],3))+','+str(np.round(test[i,12+1]+3*test[i,12+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,12+2])+ '&' +'('+str(np.round(test[i,12+2]-3*test[i,12+2+3],3))+','+str(np.round(test[i,12+2]+3*test[i,12+2+3],3))+')'+
          '\\'+'\\',file=data)
for i in range(5):
    print('&'+'$'+'C'+str(i+1)+'$'+
          '&'+str(test[i,3+0])+'&'+str(test[i,18+0])+ '&' +'('+str(np.round(test[i,18+0]-3*test[i,18+0+3],3))+','+str(np.round(test[i,18+0]+3*test[i,18+0+3],3))+')'+
          '&'+str(test[i,3+1])+'&'+str(test[i,18+1])+ '&' +'('+str(np.round(test[i,18+1]-3*test[i,18+1+3],3))+','+str(np.round(test[i,18+1]+3*test[i,18+1+3],3))+')'+
          '&'+str(test[i,3+2])+'&'+str(test[i,18+2])+ '&' +'('+str(np.round(test[i,18+2]-3*test[i,18+2+3],3))+','+str(np.round(test[i,18+2]+3*test[i,18+2+3],3))+')'+
          '\\'+'\\',file=data)
data.close()


