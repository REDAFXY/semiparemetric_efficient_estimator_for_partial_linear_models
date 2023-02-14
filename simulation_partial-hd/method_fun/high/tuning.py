'''------------------------------------------------------------------------------------------------------'''
'''3. tuning parameter function
---3.1 CV
-------KDRE: 线性情况下求β的EM算法
---3.2 BIC
---3.2-2 总：
-------LDCHOICECV: used to choose ld before estimation if necessary

---3.3 EBBS---h
--------------selectbw
---3.3-3 EBBS--总
--------------selectbwfinal
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''

import numpy as np
from method_fun.high.semi_fun import *


def BIC(u,x,y,beta0,lds,meth,hall,alpha = -0.3,typek = 'EPA',typek2 = 'Gaussian',tolcl=0.0001,tolk=0.0001,tol =0.000001,gamma = 1):
    '''wang and li for PLSE; chen for PSEMI'''
    n,p = x.shape
    BIC1 = []
    BIC = []
    mse = []
    for ld in lds:
        beta_hat, theta_hat, loglike = reg(u,x,y,beta0,ld*np.ones(6),meth,hall,alpha,typek,typek2,tolcl,tolk,tol,gamma)
        index = (beta_hat!=0).ravel()
        if((meth==0)+(meth==3) ==1):
            #BIC.append(np.log(np.mean((y-x.dot(beta_hat)-theta_hat )**2)) + DF * np.log(n)/n)
            BIC.append(np.log(np.mean((y-x.dot(beta_hat)-theta_hat )**2)) + np.log(np.log(n))/n*np.log(p)*np.sum(index))
        else:
            #BIC.append(-2*loglike + DF * np.log(n))
            BIC.append(-2*loglike + np.log(np.log(n))*np.log(p)*np.sum(index))
        BIC1.append(np.log(np.log(n))/n*np.log(p)*np.sum(index))
        beta_true=np.zeros(p).reshape(p,1)
        beta_true[0:5,0]=np.array([3,1.5,0,0,2])
        mse.append(np.mean(((beta_hat-beta_true )**2)))
    BIC = np.round(BIC,2)
    ldmax = lds[max([i for i, j in enumerate(BIC) if j == min(BIC)])]
    ldmin = lds[np.argmin(BIC)]
    ldmean = (ldmin+ldmax)/2
    return ldmean,BIC,lds[np.argmin(mse)],mse,lds[np.argmin(BIC1)],BIC1



def selectbw(u,y,x,beta0,meth,ldss,alpha = -0.3,typek = 'EPA',typek2 = 'Gaussian',tolcl=0.0001,tolk=0.0001,tol =0.000001,gamma = 1):
    n,p = x.shape
    #beta0 = np.zeros(p).reshape(p,1)
    nbs = 20
    NumPoints = 5*(n<100)+10*(n>=100)
    usort = np.sort(u.ravel())
    hmin = min(np.max(np.hstack((usort[NumPoints:]-usort[:-NumPoints]))),2.346*n**(-1/3)*np.std(u)/1.2)
    hmax = max(min((np.max(u)-np.min(u))/2,9*n**(-1/4)*np.std(u),1),2.346*n**(-1/3)*np.std(u)*1.2)
    hs = np.logspace(np.log10(hmin),np.log10(hmax),nbs).reshape(nbs,1)
    AIC = []
    for h in hs:
        beta_hat, theta_hat, loglike = reg(u,x,y,beta0,ldss,meth,h*np.ones(6),alpha,typek,typek2,tolcl,tolk,tol,gamma)
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

