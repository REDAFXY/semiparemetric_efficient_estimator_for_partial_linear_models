# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:30:01 2021

@author: public
"""
import os
import numpy as np
import numpy.linalg as la
from method_fun.high.preg_fun import *

'''------------------------------------------------------------------------------------------------------'''
'''1. estimatIon function
---1.1 低维
-------profile_lse: 基础估计，给uxy得到beta theta，相当于解了一个local linear
-------KDREtheta：SEMI的theta部分估计，是一个EM算法。
-------pKDRE：SEMI的beta部分估计，是一个EM算法
---1.2 高维
-------lasso：坐标下降
-------LLA：计算SCAD/MCP的的算法（fan zou）
------------Pld/pld：SCAD/MCP的惩罚计算
------------coordinatewlasso: 作标下降，lasso的不等权重计算
-------CD_reg:计算SCAD/MCP的CD算法（fan zou）
---1.3 总
------------reg：总的计算, meth=012,345 分别对应SCAD的lse,locaPSEMI*,PSEMI;MCP的lse,locaPSEMI*,PSEMI
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''

# from preg_fun import *



# Project
#   - folder_for_method
#   - folder_for simulationdata
#   - folder_somethingblabla
#   - folder_result1
#   - folder_result2
#   - folder_result?
#   - file_script1:generate-data - get solution - save result file
#   - file_script2
#   ...



def ke(typek,t):
    if typek=='EPA':
        r = 3/4*(1-t**2)*(abs(t)<=1)
    elif typek=='Gaussian':
        r = np.exp(-0.5*t**2)/np.sqrt(2*np.pi)
    return r
       
def profile_lse(u,y,x,h,typek ='EPA'):
    n = u.shape[0]
    S = []
    for i in range(n):
        u0 = u[i]
        t = (u - u0)/h
        D = np.hstack((np.ones(n).reshape(n,1),t))
        W = np.diag(ke(typek,t.ravel())/h)
        s = la.pinv(D.T.dot(W).dot(D)).dot(D.T).dot(W)[0]
        S.append(s)
    I = np.eye(n)
    S = np.array(S)
    beta_hat = la.pinv(x.T.dot(I-S.T).dot(I-S).dot(x)).dot(x.T).dot(I-S.T).dot(I-S).dot(y)
    theta_hat = S.dot(y-x.dot(beta_hat))
    return beta_hat,S,theta_hat



def KDREtheta(res,u,theta0,mu,hb,alpha = -0.30,typek = 'EPA', typek2 = 'Gaussian',tol =0.000001):
    n = len(res)
    if typek2 == 'Gaussian':
       h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    #h = 1.06*n**alpha*np.std(res)
    diff = 1
    loglike = -np.inf
    theta = np.hstack((theta0,np.zeros((n,1))))
    k = 0
    while (np.abs(diff)>tol) &(k<500):
        loglike_old = loglike
        loglikei = np.zeros(n)
        for i in range(n):
            G = np.hstack((np.ones(n).reshape(n,1),(u-u[i])/hb))
            khi = ke(typek,(u-u[i])/hb)/hb
            t = (mu-G.dot(theta[i,:].reshape(2,1))-res.T)/h
            pik = ke(typek2,t)/h
            #pik = np.exp(-(mu-G.dot(theta[i,:].reshape(2,1))-res.T)**2/2/h**2)/np.sqrt(2*np.pi*h**2)
            aa = np.sum(pik,1).reshape(n,1)
            aa[aa==0]=np.inf
            rik = pik/aa
            theta[i,:] = la.pinv(G.T.dot(khi*G)).dot((khi*G).T).dot(np.sum(rik*(mu-res.T),1))
            aa[aa==np.inf]=1
            loglikei[i]  = np.sum(np.log(aa)*khi)
        k = k+1
        #print(k)
        loglike = np.min(loglikei)
        #速度min>max>mean>sum
        #准确度min<max<mean=sum
        diff = loglike - loglike_old
    return theta[:,0].reshape(n,1)




def pKDRE(x,y,beta0,res,ld,ptype = 'SCAD',alpha=-0.30,typek2 = 'Gaussian',tolcl=1e-5,tolk=1e-5,gamma = 1):
    '''partial linear linear version, KDRE2 + reg2 = reg'''
    n,p=x.shape
    #third step MLE--EM  
    if typek2 == 'Gaussian':
       h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    diff, ite =1, 0
    VAL = np.inf
    ld2 = ld*h**2*2*gamma
    diff2 = 1
    while (np.abs(diff)>tolk)&(diff2>1e-6)& (ite<500):
        #print(ite,beta0[:5].reshape(1,5))
        t = (y-x.dot(beta0)-res.T)/h
        pij = ke(typek2,t)/h
        aa = np.sum(pij,1).reshape(n,1)
        aa[aa==0]=np.inf
        rho = pij/aa
        y2 = y - np.sum(rho*res.T,axis=1).reshape(n,1)
        beta1 = CD_reg(x,y2,beta0,ld2,ptype,tolcl)#M
        aa[aa==np.inf]=1
        VAL1 = -np.sum(np.log(aa))+np.sum(Pld(np.abs(beta1),ld2,ptype))
        loglike = np.sum(np.log(aa))
        diff = VAL - VAL1
        VAL = VAL1
        diff2 = np.sum((beta0-beta1)**2)
        beta0 = beta1
        ite=ite+1  
        np.abs(diff)
    # print(ite,ld2,np.sum(beta0!=0),beta0[:5].reshape(1,5))
    return beta1,loglike


def reg(u,x,y,beta0,ld,meth,hall,alpha = -0.3,typek = 'EPA',typek2 = 'Gaussian',tolcl=1e-5,tolk=1e-5,tol =1e-5,gamma = 1):
    '''
    u: n*1 vector-covariate
    x: n*p vector-covariate
    y: n*1 vector
    beta0: initial value of beta; arbitrary for plse and beta_plse for semi(*).
    meth：
        [0：lse-scad
        1：semi*-scad
        2：semi-scad
        3：lse-mcp
        4：semi*-mcp
        5：semi-mcp] 
    ld: scalar*np.ones(6)
    hall: scalar*np.ones(6)
    alpha: rate of h for kde_epsilon
    typek2: kernel type for kde_epsilon
    typek: kernel type for theta
    tolcl: tol for CD,
    tolk: tol for KDRE(estimate beta),
    tol: tol for KDREtheta(estimate theta)
    '''
    penal = ['SCAD','SCAD','SCAD','MCP','MCP','MCP']
    S = profile_lse(u,y,x,hall[meth],typek)[1]
    n,p = x.shape
    I = np.eye(n)
    xx = (I-S).dot(x)
    yy = (I-S).dot(y)
    loglike = 0
    if meth%3==0:
        beta0=CD_reg(xx,yy,beta0,ld[meth],penal[meth],tolcl)
        theta0 = S.dot(y-x.dot(beta0))
    elif meth%3 == 1:
        beta2stage = profile_lse(u,y,x[:,(beta0!=0).ravel()],hall[meth],typek)[0]
        res = (I-S).dot(y-x[:,(beta0!=0).ravel()].dot(beta2stage))
        beta0,loglike=pKDRE(xx,yy,beta0,res,ld[meth],penal[meth],alpha,typek2,tolcl,tolk,gamma)
        theta0 = S.dot(y-x.dot(beta0))
    elif meth%3 == 2:
        theta0 = S.dot(y-x.dot(beta0))
        beta2stage = profile_lse(u,y,x[:,(beta0!=0).ravel()],hall[meth],typek)[0]
        res = (I-S).dot(y-x[:,(beta0!=0).ravel()].dot(beta2stage))
        theta0 = KDREtheta(res,u,theta0,y - x.dot(beta0),hall[meth],alpha,typek, typek2,tol)
        yy = y - theta0
        beta0,loglike=pKDRE(x,yy,beta0,res,ld[meth],penal[meth],alpha,typek2,tolcl,tolk,gamma)
    return beta0,theta0,loglike


'''---------------------------------------------------------------------------------------------------'''
'''2. predict function
---2.1 thetahat: 预测 for (P)PSEMI
---2.2 thetahat_lse: 预测 for (P)SEMI* and (P)PPLSE
------------------------------------------------------------------------------------------------------'''
'''---------------------------------------------------------------------------------------------------'''


def thetahatall(thetahat,u,h,utest,typek = 'EPA'):
    fu = []
    n = len(thetahat)
    e = np.ones((n,1))
    for u0 in utest:
        Z = np.hstack((e,u-u0))
        W = np.diag((ke(typek,(u-u0)/h)/h).ravel())
        b = la.pinv(Z.T.dot(W).dot(Z)).dot(Z.T).dot(W).dot(thetahat)
        fu.append(b)
    return np.array(fu)[:,0]






