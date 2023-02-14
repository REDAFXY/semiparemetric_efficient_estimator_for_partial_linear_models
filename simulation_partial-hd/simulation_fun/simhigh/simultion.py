import os
import time
import numpy as np
from method_fun.high.tuning import *
from simulation_fun.simhigh.generate_data import *

def predsim(errortype,par_1_default,par_2_default):
    '''估计和预测的函数，得到初始结果'''
    n,p,ldmin,ldmax,nld = par_1_default
    gamma,alpha,typek,typek2,tolcl,tolk,tol = par_2_default
    beta_true=np.zeros(p).reshape(p,1)
    beta_true[0:5,0]=np.array([3,1.5,0,0,2])
    betaall = np.zeros((p,6))
    allmus = np.zeros((int(n/2),6))
    parmus = np.zeros((int(n/2),6))
    nonpamus = np.zeros((int(n/2),6))
    ldssre = np.zeros(6)
    hallre = np.zeros(6)
    q = (ldmax/ldmin)**(float(1)/nld)
    lds = ldmin*q**(np.linspace(0,nld,nld+1))
    dataall = []
    beta0 = np.zeros(p).reshape(p,1)
    x,y,u = gendata(errortype, n , p)
    dataall.append([x,y,u])
    #beta0 = np.zeros(p).reshape(p,1)
    ldss = np.zeros(6)
    hall = 1.06*n**alpha*np.ones(6)*np.std(u)
    for k in range(6):
        if(k%3==0):
            betain = beta0
        else:
            betain = reg(u,x,y,beta0,ldss,int(k//3*3),hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0]
        ldss[k] = BIC(u,x,y,betain,lds,k,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0]
        hall[k] = selectbw(u,y,x,betain,k,ldss,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0]
        ldss[k] = BIC(u,x,y,betain,lds,k,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0]
        print(k,'ld',ldss[k],'h',hall[k])
        #print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    ldssre = ldss
    hallre = hall
    xx,yy,uu = gendata(2, int(n/2) , p)
    thetaa = np.cos(2*np.pi*uu)
    for j in range(6):
        if(j%3==0):
            betain = beta0
        else:
            betain = betaall[:,j//3*3].reshape(p,1)
        beta,theta_hat,loglilke = reg(u,x,y,betain,ldss,j,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)
        betaall[:,j] = beta.ravel()
        thetaa_hat = thetahatall(theta_hat,u,hall[j],uu,typek)
        yyhat = xx.dot(beta)+thetaa_hat
        allmus[:,j] = (yy-yyhat).ravel()
        nonpamus[:,j] = (thetaa-thetaa_hat).ravel()
        parmus[:,j] = (xx.dot(beta-beta_true)).ravel()
    return betaall,parmus,nonpamus,allmus,ldssre,hallre,dataall


def predsim_diffld(ldss,hall,errortype,par_1_default,par_2_default,ratio=1):
    '''估计和预测的函数，得到初始结果'''
    n,p,ldmin,ldmax,nld = par_1_default
    gamma,alpha,typek,typek2,tolcl,tolk,tol = par_2_default
    beta_true=np.zeros(p).reshape(p,1)
    beta_true[0:5,0]=np.array([3,1.5,0,0,2])
    betaall = np.zeros((p,6))
    allmus = np.zeros((int(n/2),6))
    parmus = np.zeros((int(n/2),6))
    nonpamus = np.zeros((int(n/2),6))
    beta0 = np.zeros(p).reshape(p,1)
    dataall = []
    ldss = ldss*ratio
    x,y,u = gendata(errortype, n , p)
    xx,yy,uu = gendata(errortype, int(n/2) , p)
    thetaa = np.cos(2*np.pi*uu)
    for j in range(6):
        if(j%3==0):
            betain = reg(u,x,y,beta0,ldss[(j//3)*3]/ratio*np.ones(6),int(j//3)*3,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0].copy()
        else:
            betain = reg(u,x,y,beta0,ldss[(j//3)*3]/ratio*np.ones(6),int(j//3)*3,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)[0].copy()
        beta,theta_hat,loglilke = reg(u,x,y,betain,ldss,j,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)
        betaall[:,j] = beta.ravel()
        thetaa_hat = thetahatall(theta_hat,u,hall[j],uu,typek)
        yyhat = xx.dot(beta)+thetaa_hat
        allmus[:,j] = (yy-yyhat).ravel()
        nonpamus[:,j] = (thetaa-thetaa_hat).ravel()
        parmus[:,j] = (xx.dot(beta-beta_true)).ravel()
    return betaall,parmus,nonpamus,allmus,ldss,hall,dataall
