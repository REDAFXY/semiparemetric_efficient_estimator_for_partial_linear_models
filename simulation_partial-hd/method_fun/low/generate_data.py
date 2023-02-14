# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:23:57 2021

@author: public
"""
import numpy as np
from numpy.linalg import cholesky

def gendata(errortype, n , p):
    '''生成数据'''
    sigma=[0.5**np.abs(i-j) for i in range(0,p) for j in range(0,p)]
    sigma=np.array(sigma).reshape(p,p)
    R = cholesky(sigma)
    beta_true=np.zeros(p).reshape(p,1)
    beta_true[0:5,0]=np.array([3,1.5,0,0,2])
    x = np.dot(np.random.randn(n, p), R)
    u = np.random.rand(n,1)
    theta = np.cos(2*np.pi*u)
    if errortype==0:
        error = np.random.randn(n,1)
    elif errortype==1:
        temp = np.random.binomial(1,0.3,n)
        error = (np.random.randn(n,1)-1.4)*temp.reshape(n,1)+(np.random.randn(n,1)*0.4+0.6)*(1-temp).reshape(n,1)
    elif errortype==2:
        error = (np.random.standard_t(3,n)/np.sqrt(3)).reshape(n,1)
    elif errortype==3:
        temp = np.random.binomial(1,0.95,n)
        error = np.random.randn(n,1)*0.85*temp.reshape(n,1)+np.random.randn(n,1)*2.5*(1-temp).reshape(n,1)
    elif errortype==4:
        temp = np.random.binomial(1,0.5,n)
        error = (np.random.randn(n,1)*0.5-1)*temp.reshape(n,1)+(np.random.randn(n,1)*0.5+1)*(1-temp).reshape(n,1)
    y = theta + x.dot(beta_true) + error
    return x,y,u