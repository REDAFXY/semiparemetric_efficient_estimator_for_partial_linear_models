# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:33:17 2021

@author: public
"""
'''------------------------------------------------------------------------------------------------------
---1.2 高维
-------lasso：坐标下降
-------LLA：计算SCAD/MCP的的算法（fan zou）
------------Pld/pld：SCAD/MCP的惩罚计算
------------coordinatewlasso: 作标下降，lasso的不等权重计算
-------CD_reg:计算SCAD/MCP的CD算法（fan zou）
------------------------------------------------------------------------------------------------------'''
import numpy as np

'''LASSO'''
def lasso(X,y,beta0,ld,toll=0.0001):
    n,p = X.shape
#    beta = la.inv(X.T.dot(X)).dot(X.T).dot(y)
    beta = beta0.copy()
    diff = 1
    ite=0
    while (np.abs(diff) > toll) & (ite<1000):
        VAL = np.sum((y - X.dot(beta))**2)/(2*n)+ ld*np.sum(np.abs(beta))
        for j in range(p):
            beta[j] = 0
            y2 = X.dot(beta)
            t = X[:,j].dot(y-y2)
            beta[j] =np.sign(t)*(np.abs(t)-n*ld)*((np.abs(t)-n*ld)>0)/(X[:,j].dot(X[:,j].T))
        VAL2 = np.sum((y - X.dot(beta))**2)/(2*n)+ ld*np.sum(np.abs(beta))
        diff = VAL2 - VAL
        VAL = VAL2
        ite+=1
#        print('lasso',diff,ite)
    return beta


'''method0 SCAD'''    
def Pld(t,ld,type = 'SCAD',a=3.7):
    if type == 'SCAD':
        a=3.7
        re=ld*np.abs(t)*(np.abs(t)<ld)+ld**2*(a+1)/2*(np.abs(t)>a*ld)+(2*ld*a*np.abs(t)-t**2-ld**2)/2/(a-1)*((np.abs(t)>=ld)&(np.abs(t)<=a*ld))
    elif type == 'MCP':
        a=3
        re=(ld*np.abs(t)-t**2/(2*a))*(np.abs(t)<=a*ld)+(a*ld**2/2)*(np.abs(t)>a*ld)
    return re

def pld(t,ld,type = 'SCAD',a=3.7):
    if type == 'SCAD':
        a=3.7
        re=ld*(t<=ld)+(a*ld-t)*(a*ld>t)/(a-1)*(t>ld)
    elif type == 'MCP':
        a=3
        re=(ld-np.abs(t)/a)*(np.abs(t)<a*ld)
    return re


def coordinatewlasso(x,y,beta0,w,tolcl=0.0001):
    n,p = x.shape
    beta = beta0.copy()
    diff = 1
    VAL = np.sum((y - x.dot(beta))**2)/(2*n) + np.abs(beta).T.dot(w)
    ite = 0
    while (np.abs(diff) > tolcl) & (ite<1000):
        ite=ite+1
        for k in range(p):
            beta[k] = 0
            y2 = x.dot(beta).reshape(n,1)
            t = x[:,k].dot(y-y2)
            beta[k] = np.sign(t)*(np.abs(t)-n*w[k])*((np.abs(t)-n*w[k])>0)/(x[:,k].dot(x[:,k]))
        VAL2 = np.sum((y - x.dot(beta))**2)/(2*n) + np.abs(beta).T.dot(w)
        diff = VAL2 - VAL
        VAL = VAL2
#        print('coolasso',ite,diff)
    return beta



def LLA(x,y2,beta0,ld,ptype='SCAD',tolcl=0.0001): 
    n,p=x.shape
    #给定LLA的初值beta0，一般是初值为betalasso，初值为0也行其实ld大的话0就是lasso
    #1，w的penalty
    w=pld(np.abs(beta0),ld,ptype)
    #2，更新beta和w
    beta0 = coordinatewlasso(x,y2,beta0,w,tolcl)
    w=pld(np.abs(beta0),ld,ptype)
    beta0 = coordinatewlasso(x,y2,beta0,w,tolcl)
    return beta0

def CD_reg(X,y,w_arr,lamb,method= 'SCAD',tolcl=1e-5):
    n,p = np.shape(X)
    w_arr_new = np.zeros((p,1))
    flag = 1
    if method == 'SCAD':
        gamma = 3.7
    if method == 'MCP':
        gamma = 3
#    tau = 0.1#(tlp的参数)
    diff = 1
    VAL = np.sum((y - X.dot(w_arr))**2)/(2*n) + np.sum(Pld(np.abs(w_arr),lamb,method))
    while (np.abs(diff) > tolcl) & (flag<1000):
        #rho = 1000/n*int(flag/50+1)*5 # 这个参数类似于 学习率 可以定死 也可以 随着n变化而变化 .rho越大 迭代次数越多 但是对 sum(w_arr） = 1 就越好
        for i in range(p):  
            if method == 'lasso':
                hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n - lamb
                w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)
            if method == 'SCAD':
                if w_arr[i] < lamb:
                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n - lamb
                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)
                elif w_arr[i] > gamma*lamb:
                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n
                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)
                else:
                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n  - lamb*gamma/(gamma-1)
                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n - 1/(gamma-1))
            if method == 'MCP':
                if w_arr[i] <  gamma*lamb:
                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n  - lamb
                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n - 1/gamma)
                else :
                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n
                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)  
#            if method == 'tlp':
#                if w_arr[i] <  tau:
#                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n  - lamb/tau
#                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)
#                else :
#                    hhhh = (X[:,i]).reshape(n,1).T.dot(y - X.dot(w_arr) + (X[:,i]*w_arr[i]).reshape(n,1))/n 
#                    w_arr_new[i] = (hhhh>0)*hhhh / (X[:,i].T.dot(X[:,i])/n)                
        VAL2 = np.sum((y - X.dot(w_arr_new))**2)/(2*n) + np.sum(Pld(np.abs(w_arr_new),lamb,method))
        diff = VAL2 - VAL
        VAL = VAL2
        w_arr = w_arr_new 
        flag += 1
        #print(flag)
    return w_arr