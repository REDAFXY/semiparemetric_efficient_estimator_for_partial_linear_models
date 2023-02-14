import os
import time
import numpy as np
from method_fun.low.semi_func import *
from simulation_fun.simlow.generate_data import *
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
    ---注：用上一段（predsim）和下一段（KDRESIMsemi）都行，这里用的是下一段（KDRESIMsemi）。
---4.3 table2-SD
  ---simulation
    ---sdestlse/KDRESIMlse: 估计lse方差的函数/simulation函数【最后没用其实】
    ---sdestsemi/KDRESIMsemi: 估计semi方差的函数/simulation函数
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''




def sdestsemi(x,y,u,alpha=-0.30,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5):
    n,p = x.shape
    #estimate
    hall=np.ones(3)
    for k in range(2):
        print(time.strftime('%H.%M.%S',time.localtime(time.time())))
        if(k==0):
            betain,Sin = 0,0
        else:
            betain,Sin,thetain,loglike = reg(u,x,y,betain,Sin,0,hall,alpha,typek,typek2)
        hall[k] = selectbw(u,y,x,betain,Sin,k,alpha,typek,typek2)[0]
        print(k,'h',hall[k])
    hall[2] = hall[1]
    beta0 = betain
    betakdre,Skdre,thetakdre,loglike = reg(u,x,y,betain,Sin,2,hall,alpha,typek,typek2)
    betakdrestar,Skdrestar,thetakdrestar,loglike = reg(u,x,y,betain,Sin,1,hall,alpha,typek,typek2)
    res = y - thetain - x.dot(betain)
    if typek2 == 'Gaussian':
        h = 1.06*n**alpha*np.std(res)
    elif typek2 == 'EPA': 
        h = 2.346*n**alpha*np.std(res)
    elif typek2 == 'BW': 
        h = 2.78*n**alpha*np.std(res)
    #要改SD的hat
    t=(y-x.dot(betakdre)-thetakdre-res.T)/h      
    eta = np.sum(ke(t,typek2),axis = 1)/h
    etad = np.sum(ke_dot(t,typek2),axis = 1)/h**2
    etadd = np.sum(ke_ddot(t,typek2),axis = 1)/h**3
    logetad = etad/eta
    logetadd = (etadd*eta-etad**2)/eta**2
    meanx = np.mean(x, axis = 0).reshape(p,1)
    ##SEMI
    xx = x-meanx.T
    ##(1)(2)SEMI-直接算
    Ihat = xx.T.dot(np.diag(logetadd)).dot(xx)/n
    Ihat2 = xx.T.dot(np.diag(logetad**2)).dot(xx)/n
    sdhat = np.sqrt(-np.diag(la.pinv(Ihat))/n)
    sdhat2 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
    ##(3)SEMI-独立算
    Ihat = -np.cov(xx.T.dot(np.diag(logetad)))
    ##(4)lse
    Ihat2 = xx.T.dot(xx)/n
    sdhat3 = np.sqrt(-np.diag(la.pinv(Ihat))/n)
    sdhat4 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
    ##SEMI*
    ##SEMI*-直接算cov
    epsilon = (y -thetakdrestar - x.dot(betakdrestar)).ravel()
    Fstar = xx.T.dot(np.diag(logetadd)).dot(x)/n
    xlogetaddmean = np.mean(x.T.dot(np.diag(logetadd)),1).reshape(p,1)
    Sigmastar = np.cov(x.T.dot(np.diag(logetad))-xlogetaddmean.dot(np.ones((1,n))).dot(np.diag(epsilon)))
    tmp = la.pinv(Fstar).dot(Sigmastar).dot(la.pinv(Fstar))
    sd_es = np.sqrt(np.diag(tmp)/n) 
    ##SEMI*-算化简的式子
    Ihat2 = x.T.dot(np.diag(logetad**2)).dot(x)/n
    Sigmastar2 = Ihat2 + meanx.dot(meanx.T)*(-2*np.mean(logetadd)*(np.mean(logetad*epsilon)-np.mean(epsilon)*np.mean(logetad))-np.mean(logetad**2)+np.mean(logetadd)**2*np.var(epsilon))
    tmp = la.pinv(Fstar).dot(Sigmastar2).dot(la.pinv(Fstar))
    sd_es2 = np.sqrt(np.diag(tmp)/n)
    ##SEMI*-直接算cov--用Ihat2代替Fstar
    Ihat2 = xx.T.dot(np.diag(logetad**2)).dot(xx)/n
    #Ihat = xx.T.dot(np.diag(logetadd)).dot(xx)/n
    tmp = la.pinv(Ihat2).dot(Sigmastar2).dot(la.pinv(Ihat2))
    sd_es3 = np.sqrt(np.diag(tmp)/n) 
    ##SEMI*-算化简的式子--用Ihat2代替Fstar
    tmp = la.pinv(Ihat2).dot(Sigmastar).dot(la.pinv(Ihat2))
    sd_es4 = np.sqrt(np.diag(tmp)/n) 
    betakdreall = betakdre.ravel()
    betakdreallstar = betakdrestar.ravel()
    betakdrealllse = beta0.ravel()
    thetahat = np.hstack((thetain,thetakdre,thetakdrestar))
    return sdhat,sdhat2,sdhat3,sdhat4,sd_es,sd_es2,sd_es3,sd_es4,betakdreall,betakdreallstar,betakdrealllse,thetahat,hall


def simlowall(ntimes,errortype,size,alpha=-0.30,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5):
    p=3
    n=size
    sdhat = np.zeros((ntimes,3))
    sdhat2 = np.zeros((ntimes,3))
    sdhat3 = np.zeros((ntimes,3))
    sdhat4 = np.zeros((ntimes,3))
    sdhat5 = np.zeros((ntimes,3))
    sdhat6 = np.zeros((ntimes,3))
    sdhat7 = np.zeros((ntimes,3))
    sdhat8 = np.zeros((ntimes,3))
    thetaall = np.zeros((ntimes,n,3))
    data = []
    beta_true=np.array([3,1.5,2]).reshape(3,1)
    hallre = np.zeros((ntimes,3))
    betaall = np.zeros((ntimes,p,3))
    allmus = np.zeros((ntimes,int(n/2),3))
    parmus = np.zeros((ntimes,int(n/2),3))
    nonpamus = np.zeros((ntimes,int(n/2),3))
    for i in range(ntimes):
        x,y,u = gendata(errortype, n , p, beta_true)
        #beta0 = np.zeros(p).reshape(p,1)
        data.append([x,y,u])
        hall = np.ones(3)*2.346*n**alpha*np.std(u)
        sdhat[i,:],sdhat2[i,:],sdhat3[i,:],sdhat4[i,:],sdhat5[i,:],sdhat6[i,:],sdhat7[i,:],sdhat8[i,:],betaall[i,:,2],betaall[i,:,1],betaall[i,:,0] ,thetaall[i,:,:],hallre[i,:] =  sdestsemi(x,y,u,alpha,typek,typek2,tolk)
        xx,yy,uu = gendata(errortype, int(n/2) , p, beta_true)
        thetaa = np.cos(2*np.pi*uu)
        for j in range(3):
            betahat = betaall[i,:,j].reshape(p,1)
            thetahat = thetaall[i,:,j].reshape(n,1)
            thetaa_hat = thetahatall(thetahat,u,hall[j],uu,typek)
            yyhat = xx.dot(betahat)+thetaa_hat
            allmus[i,:,j] = (yy-yyhat).ravel()
            nonpamus[i,:,j] = (thetaa-thetaa_hat).ravel()
            parmus[i,:,j] = (xx.dot(betahat-beta_true)).ravel()
        print('time',i)
        print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    return [betaall,parmus,nonpamus,allmus,hallre,data],[sdhat,sdhat2,sdhat3,sdhat4,sdhat5,sdhat6,sdhat7,sdhat8,betaall,thetaall]






