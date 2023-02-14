import numpy as np
import numpy.linalg as la
#import matplotlib.pyplot as plt
from numpy.linalg import cholesky


def lse(x,y):
    beta = la.pinv(x.T.dot(x)).dot(x.T).dot(y)
    eps = y-x.dot(beta)
    return beta,eps

def ke(t,typeke = 'EPA'):
    if typeke == 'Gaussian':
        re = np.exp(-0.5*t**2)/np.sqrt(2*np.pi)
    elif typeke == 'EPA':
        re = 3/4*(1-t**2)*(abs(t)<=1)
    elif typeke == 'BW':
        re = 15/16*(1-t**2)**2*(abs(t)<=1)
    else:
        re ='no such kernel'
    return re

          
            
def ke_dot(t,typeke = 'EPA'):
    if typeke == 'Gaussian':
        re = np.exp(-0.5*t**2)/np.sqrt(2*np.pi)*(-t)
    elif typeke == 'EPA':
        re = -3/2*t*(abs(t)<=1)
    elif typeke == 'BW':
        re = -4*t*15/16*(1-t**2)*(abs(t)<=1)
    else:
        re ='no such kernel'
    return re


def ke_ddot(t,typeke = 'EPA'):
    if typeke == 'Gaussian':
        re = np.exp(-0.5*t**2)/np.sqrt(2*np.pi)*(t**2-1)
    elif typeke == 'EPA':
        re = -3/2*(abs(t)<=1)
    elif typeke == 'BW':
        re = -4*15/16*(1-3*t**2)*(abs(t)<=1)
    else:
        re ='no such kernel'
    return re


def profile_lse(u,y,x,h,typeke = 'EPA'):
    n,p = x.shape
    e = np.ones((n,1))
    I = np.eye(n)
    S = np.zeros((n,n))
    for i in range(n):
        u0 = u[i]
        Z = np.hstack((e,u-u0))
        W = np.diag((ke((u-u0)/h,typeke)/h).ravel())
        S[i,:] = (la.pinv(Z.T.dot(W).dot(Z)).dot(Z.T).dot(W))[0]
    beta_hat = la.pinv(x.T.dot(I-S.T).dot(I-S).dot(x)).dot(x.T).dot(I-S.T).dot(I-S).dot(y)
    theta_hat = S.dot(y-x.dot(beta_hat))
    loglike = 0
    return beta_hat,S,theta_hat,loglike




'''kdrebasic'''
def KDRE(x,y,res,beta0,alpha=-0.30,typek2 = 'EPA',tolk = 1e-5):
    n,p=x.shape
    #h = 1.06*n**alpha*np.std(res)
    if typek2 == 'Gaussian':
       h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    elif typek2 == 'BW': 
        h = 2.78*n**alpha*np.std(res)
    VAL0,diff = np.inf,np.inf
    ite  = 0
    while np.abs(diff)>tolk:
        rhoij = ke((y-x.dot(beta0)-res.T)/h,typek2)/h
        VAL = np.sum(np.log(np.mean(rhoij, axis = 1)))
        rhoij = rhoij/np.sum(rhoij, axis = 1).reshape(n,1)
        yy = np.sum(rhoij*(y-res.T), axis = 1).reshape(n,1)
        beta0 = lse(x,yy)[0]
        diff = VAL -VAL0
        ite=ite+1 
        VAL0 = VAL
        #print(ite,VAL0)
    return beta0,VAL

def KDRE2(x,y,res,beta0,alpha=-0.30,typek2 = 'Gaussian',tolk=1e-5):
    '''partial linear linear version, KDRE2 + reg2 = reg'''
    n,p=x.shape
    #third step MLE--EM  
    if typek2 == 'Gaussian':
        h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    diff, ite =1, 0
    VAL = np.inf
    diff2 = 1
    while (np.abs(diff)>tolk)&(diff2>1e-6):
        #print(ite,beta0[:5].reshape(1,5))
        t = (y-x.dot(beta0)-res.T)/h
        pij = ke(typek2,t)/h
        aa = np.sum(pij,1).reshape(n,1)
        aa[aa==0]=np.inf
        rho = pij/aa
        y2 = y - np.sum(rho*res.T,axis=1).reshape(n,1)
        beta1 = lse(x,y2)#M
        aa[aa==np.inf]=1
        VAL1 = -np.sum(np.log(aa))
        loglike = -VAL1
        diff = VAL - VAL1
        VAL = VAL1
        diff2 = np.sum((beta0-beta1)**2)
        beta0 = beta1
        ite=ite+1  
        np.abs(diff)
    return beta1,loglike

                  
def profile_lseKDRE(u,y,x,beta0,S,h,alpha=-0.30,typek ='EPA',typek2 ='EPA',tolk=0.000001):
    n,p=x.shape
    #first step: LSE
    #second step: KDE（实际上只是算出了KDE的样本点）
    I = np.eye(n)
    res = (I-S).dot(y-x.dot(beta0))
    #third step Maximize
    xx = (I-S).dot(x)
    yy = (I-S).dot(y)
    beta1,loglike = KDRE(xx,yy,res,beta0,alpha,typek2,tolk)
    theta_hat = S.dot(y-x.dot(beta1))
    return beta1,S,theta_hat,loglike



'''---------------------------------------------------------------------------------leKDRE'''

'''theta-basic'''

def KDREtheta(res,u,theta0,mu,hb,alpha = -0.30,typek = 'EPA', typek2 = 'EPA',tol =1e-5):
    n = len(res);
    if typek2 == 'Gaussian':
       h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    elif typek2 == 'BW': 
        h = 2.78*n**alpha*np.std(res)
    #h = 1.06*n**alpha*np.std(res)
    diff = 1
    loglike = -np.inf
    theta = np.hstack((theta0,np.zeros((n,1))))
    k = 0
    loglikeall = []
    while (np.abs(diff)>tol) &(k<500):
        loglike_old = loglike
        loglikei = np.zeros(n)
        for i in range(n):
            G = np.hstack((np.ones(n).reshape(n,1),(u-u[i])/hb))
            khi = ke((u-u[i])/hb,typek)/hb
            t = (mu-G.dot(theta[i,:].reshape(2,1))-res.T)/h
            pik = ke(t,typek2)/h
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
        loglikeall.append(loglike)
    return theta[:,0].reshape(n,1)



def backfitting_KDRE(u,y,x,beta0,S,hb,alpha=-0.30,typek = 'EPA', typek2 = 'EPA',tol=0.000001,tolk=0.000001):
    n,p=x.shape
    #first step: LSE
    #second step: KDE（实际上只是算出了KDE的样本点）
    I = np.eye(n)
    res = (I-S).dot(y-x.dot(beta0))
    #third step MLE
    theta0 = S.dot(y-x.dot(beta0))
    theta1= KDREtheta(res,u,theta0,y - x.dot(beta0),hb,alpha,typek, typek2,tol)
    y2 = y - theta1
    beta1,loglike = KDRE(x,y2,res,beta0,alpha,typek2,tolk)
    return beta1,S,theta1,loglike



def reg(u,x,y,beta0,S,meth,hall,alpha=-0.30,typek ='EPA',typek2 ='EPA'):
    if meth==0:
        re = profile_lse(u,y,x,hall[0],typek)
    elif meth==1:
        re = profile_lseKDRE(u,y,x,beta0,S,hall[1],alpha,typek,typek2)
    elif meth==2:
        re = backfitting_KDRE(u,y,x,beta0,S,hall[2],alpha,typek,typek2)
    return re
    
  



def selectbw(u,y,x,beta0,S,meth,alpha=-0.3,typek ='EPA',typek2 ='EPA'):
    n = u.shape[0]
    nbs = 20
    NumPoints = 5*(n<100)+10*(n>=100)
    usort = np.sort(u.ravel())
    hmin = min(np.max(np.hstack((usort[NumPoints:]-usort[:-NumPoints]))),2.346*n**(-1/3)*np.std(u)/1.2)
    hmax = max(min((np.max(u)-np.min(u))/2,9*n**(-1/4)*np.std(u),1),2.346*n**(-1/3)*np.std(u)*1.2)
    hs = np.logspace(np.log10(hmin),np.log10(hmax),nbs).reshape(nbs,1)
    AIC = []
    for h in hs:
        beta_hat, S, theta_hat, loglike = reg(u,x,y,beta0,S,meth,h*np.ones(3),alpha,typek,typek2)
        index = (beta_hat!=0).ravel()
        if((meth==0)+(meth==3) ==1):
            #BIC.append(np.log(np.mean((y-x.dot(beta_hat)-theta_hat )**2)) + DF * np.log(n)/n)
            AIC.append(np.log(np.mean((y-x.dot(beta_hat)-theta_hat )**2)) + 2*np.sum(index))
        else:
            #BIC.append(-2*loglike + DF * np.log(n))
            AIC.append(-2*loglike + 2*np.sum(index))
    index = np.argmin(np.array(AIC))
    h_best  = hs[index]
    return h_best,np.array(AIC)

'''function for prediction'''


# def thetahatall(x,y,u,utest,beta,h,typek = 'EPA'):
def thetahatall(thetahat,u,h,utest,typeke = 'EPA'):
    fu = []
    n = len(thetahat)
    e = np.ones((n,1))
    for u0 in utest:
        Z = np.hstack((e,u-u0))
        W = np.diag((ke((u-u0)/h,typeke)/h).ravel())
        b = la.pinv(Z.T.dot(W).dot(Z)).dot(Z.T).dot(W).dot(thetahat)
        fu.append(b)
    return np.array(fu)[:,0]