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
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


# def predsim(errortype,n,par_default):
#     p=3
#     alpha, typek, typek2 = par_default
#     beta_true=np.array([3,1.5,2]).reshape(3,1)
#     hallre = np.zeros((3))
#     betaall = np.zeros((p,3))
#     allmus = np.zeros((int(n/2),3))
#     parmus = np.zeros((int(n/2),3))
#     nonpamus = np.zeros((int(n/2),3))
#     dataall = []
#     x,y,u = gendata(errortype, n , p, beta_true)
#     #beta0 = np.zeros(p).reshape(p,1)
#     dataall.append([x,y,u])
#     hall = np.ones(3)*2.346*n**alpha*np.std(u)
#     #hall = hallreall[errortype,:]
#     for k in range(3):
#         if(k==0):
#             betain,Sin = 0,0
#         else:
#             betain,Sin,thetain,loglike = reg(u,x,y,betain,Sin,0,hall,alpha,typek,typek2)
#         hall[k] = selectbw(u,y,x,betain,Sin,k,alpha,typek,typek2)[0]
#         print(k,'h',hall[k])
#     hallre = hall.copy()
#     xx,yy,uu = gendata(errortype, int(n/2) , p, beta_true)
#     thetaa = np.cos(2*np.pi*uu)
#     for j in range(3):
#         if(j==0):
#             betain,Sin = 0,0
#         else:
#             betain,Sin,thetain,loglike = reg(u,x,y,betain,Sin,0,hall,alpha,typek,typek2)
#         beta,S,thetahat,loglike = reg(u,x,y,betain,Sin,j,hall,alpha,typek,typek2)
#         betaall[:,j] = beta.ravel()
#         thetaa_hat = thetahatall(thetahat,u,hall[j],uu,typek)
#         yyhat = xx.dot(beta)+thetaa_hat
#         allmus[:,j] = (yy-yyhat).ravel()
#         nonpamus[:,j] = (thetaa-thetaa_hat).ravel()
#         parmus[:,j] = (xx.dot(beta-beta_true)).ravel()
#     return betaall,parmus,nonpamus,allmus,hallre,dataall


def predsim(ntimes,errortype,size,alpha=-0.30,typek ='EPA',typek2 ='Gaussian'):
    p=3
    n=size
    beta_true=np.array([3,1.5,2]).reshape(3,1)
    hallre = np.zeros((ntimes,3))
    betaall = np.zeros((ntimes,p,3))
    allmus = np.zeros((ntimes,int(n/2),3))
    parmus = np.zeros((ntimes,int(n/2),3))
    nonpamus = np.zeros((ntimes,int(n/2),3))
    dataall = []
    for i in range(ntimes):
        x,y,u = gendata(errortype, n , p, beta_true)
        #beta0 = np.zeros(p).reshape(p,1)
        dataall.append([x,y,u])
        hall = np.ones(3)*2.346*n**alpha*np.std(u)
        #hall = hallreall[errortype,:]
        for k in range(3):
            if(k==0):
                betain,Sin = 0,0
            else:
                betain,Sin,thetain,loglike = reg(u,x,y,betain,Sin,0,hall,alpha,typek,typek2)
            hall[k] = selectbw(u,y,x,betain,Sin,k,alpha,typek,typek2)[0]
            print(k,'h',hall[k])
        hallre[i,:] = hall.ravel()
        xx,yy,uu = gendata(errortype, int(n/2) , p, beta_true)
        thetaa = np.cos(2*np.pi*uu)
        for j in range(3):
            if(j==0):
                betain,Sin = 0,0
            else:
                betain,Sin,thetain,loglike = reg(u,x,y,betain,Sin,0,hall,alpha,typek,typek2)
            beta,S,thetahat,loglike = reg(u,x,y,betain,Sin,j,hall,alpha,typek,typek2)
            betaall[i,:,j] = beta.ravel()
            thetaa_hat = thetahatall(thetahat,u,hall[j],uu,typek)
            yyhat = xx.dot(beta)+thetaa_hat
            allmus[i,:,j] = (yy-yyhat).ravel()
            nonpamus[i,:,j] = (thetaa-thetaa_hat).ravel()
            parmus[i,:,j] = (xx.dot(beta-beta_true)).ravel()
        print('time',i)
        print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    return betaall,parmus,nonpamus,allmus,hallre,dataall
'''------------------------------------------------------------------------------------------------------'''
'''
---4.2 table1-mse/mad:
  ---simulation
    ---KDRESIMsemi
    ---?????????????????????predsim??????????????????KDRESIMsemi???????????????????????????????????????KDRESIMsemi??????
---4.3 table2-SD
  ---simulation
    ---sdestlse/KDRESIMlse: ??????lse???????????????/simulation??????????????????????????????
    ---sdestsemi/KDRESIMsemi: ??????semi???????????????/simulation??????
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


# def sdestlse(u,x,y,alpha=-0.30,typek ='EPA',typek2 ='EPA',tolk = 1e-5):
#     n,p = x.shape
#     betain,Sin = 0,0
#     h = selectbw(u,y,x,betain,Sin,0,alpha,typek,typek2)[0]
#     beta0,S,theta0,loglike = reg(u,x,y,0,h*np.ones(3),alpha,typek,typek2)
#     eps = y - theta0 - x.dot(beta0)
#     logetad = -eps.ravel()
#     logetadd = -np.ones(n)
#     meanx = np.mean(x, axis = 0).reshape(p,1)
#     xx = x-meanx.T
#     Ihat = xx.T.dot(np.diag(logetadd)).dot(xx)/n
#     Ihat2 = xx.T.dot(np.diag(logetad**2)).dot(xx)/n
#     sdhat = np.sqrt(-np.diag(la.pinv(Ihat))/n)
#     sdhat2 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
#     Ihat3 = xx.T.dot(xx)/n*(np.mean(logetadd))
#     Ihat4 = xx.T.dot(xx)/n*(np.mean(logetad**2))
#     sdhat3 = np.sqrt(-np.diag(la.pinv(Ihat3))/n)
#     sdhat4 = np.sqrt(np.diag(la.pinv(Ihat4))/n)
#     sdhat5 = np.sqrt(np.diag(la.pinv(Ihat).dot(Ihat2).dot(la.pinv(Ihat)))/n)
#     #sdhat4 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
#     betakdreall = beta0.ravel()
#     return sdhat,sdhat2,sdhat3,sdhat4,sdhat5,betakdreall

# def KDRESIMlse(ntimes,errortype,size,alpha=-0.30,typek ='EPA',typek2 ='EPA',tolk = 1e-5):
#     p=3
#     n=size
#     beta=np.array([3,1.5,2]).reshape(3,1)
#     sdhat = np.zeros((ntimes,3))
#     sdhat2 = np.zeros((ntimes,3))
#     sdhat3 = np.zeros((ntimes,3))
#     sdhat4 = np.zeros((ntimes,3))
#     sdhat5 = np.zeros((ntimes,3))
#     betakdreall = np.zeros((ntimes,3))
#     #x = gendata21(errortype,n,p,beta)
#     for i in range(ntimes):
#         x,y,u = gendata(errortype,n,p,beta)
#         sdhat[i,:],sdhat2[i,:],sdhat3[i,:],sdhat4[i,:],sdhat5[i,:],betakdreall[i,:] =  sdestlse(u,x,y,alpha,typek,typek2,tolk)
#         print(i)
#     return sdhat,sdhat2,sdhat3,sdhat4,sdhat5,betakdreall



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
    #??????SD???hat
    t=(y-x.dot(betakdre)-thetakdre-res.T)/h      
    eta = np.sum(ke(t,typek2),axis = 1)/h
    etad = np.sum(ke_dot(t,typek2),axis = 1)/h**2
    etadd = np.sum(ke_ddot(t,typek2),axis = 1)/h**3
    logetad = etad/eta
    logetadd = (etadd*eta-etad**2)/eta**2
    meanx = np.mean(x, axis = 0).reshape(p,1)
    ##SEMI
    xx = x-meanx.T
    ##(1)(2)SEMI-?????????
    Ihat = xx.T.dot(np.diag(logetadd)).dot(xx)/n
    Ihat2 = xx.T.dot(np.diag(logetad**2)).dot(xx)/n
    sdhat = np.sqrt(-np.diag(la.pinv(Ihat))/n)
    sdhat2 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
    ##(3)SEMI-?????????
    Ihat = -np.cov(xx.T.dot(np.diag(logetad)))
    ##(4)lse
    Ihat2 = xx.T.dot(xx)/n
    sdhat3 = np.sqrt(-np.diag(la.pinv(Ihat))/n)
    sdhat4 = np.sqrt(np.diag(la.pinv(Ihat2))/n)
    ##SEMI*
    ##SEMI*-?????????cov
    epsilon = (y -thetakdrestar - x.dot(betakdrestar)).ravel()
    Fstar = xx.T.dot(np.diag(logetadd)).dot(x)/n
    xlogetaddmean = np.mean(x.T.dot(np.diag(logetadd)),1).reshape(p,1)
    Sigmastar = np.cov(x.T.dot(np.diag(logetad))-xlogetaddmean.dot(np.ones((1,n))).dot(np.diag(epsilon)))
    tmp = la.pinv(Fstar).dot(Sigmastar).dot(la.pinv(Fstar))
    sd_es = np.sqrt(np.diag(tmp)/n) 
    ##SEMI*-??????????????????
    Ihat2 = x.T.dot(np.diag(logetad**2)).dot(x)/n
    Sigmastar2 = Ihat2 + meanx.dot(meanx.T)*(-2*np.mean(logetadd)*(np.mean(logetad*epsilon)-np.mean(epsilon)*np.mean(logetad))-np.mean(logetad**2)+np.mean(logetadd)**2*np.var(epsilon))
    tmp = la.pinv(Fstar).dot(Sigmastar2).dot(la.pinv(Fstar))
    sd_es2 = np.sqrt(np.diag(tmp)/n)
    ##SEMI*-?????????cov--???Ihat2??????Fstar
    Ihat2 = xx.T.dot(np.diag(logetad**2)).dot(xx)/n
    #Ihat = xx.T.dot(np.diag(logetadd)).dot(xx)/n
    tmp = la.pinv(Ihat2).dot(Sigmastar2).dot(la.pinv(Ihat2))
    sd_es3 = np.sqrt(np.diag(tmp)/n) 
    ##SEMI*-??????????????????--???Ihat2??????Fstar
    tmp = la.pinv(Ihat2).dot(Sigmastar).dot(la.pinv(Ihat2))
    sd_es4 = np.sqrt(np.diag(tmp)/n) 
    betakdreall = betakdre.ravel()
    betakdreallstar = betakdrestar.ravel()
    betakdrealllse = beta0.ravel()
    thetahat = np.hstack((thetain,thetakdre,thetakdrestar))
    return sdhat,sdhat2,sdhat3,sdhat4,sd_es,sd_es2,sd_es3,sd_es4,betakdreall,betakdreallstar,betakdrealllse,thetahat,hall


def KDRESIMsemi(ntimes,errortype,size,alpha=-0.30,typek ='EPA',typek2 ='Gaussian',tolk = 1e-5):
    p=3
    n=size
    beta=np.array([3,1.5,2]).reshape(3,1)
    sdhat = np.zeros((ntimes,3))
    sdhat2 = np.zeros((ntimes,3))
    sdhat3 = np.zeros((ntimes,3))
    sdhat4 = np.zeros((ntimes,3))
    sdhat5 = np.zeros((ntimes,3))
    sdhat6 = np.zeros((ntimes,3))
    sdhat7 = np.zeros((ntimes,3))
    sdhat8 = np.zeros((ntimes,3))
    betakdreall = np.zeros((ntimes,3))
    betakdreallstar = np.zeros((ntimes,3))
    betakdrealllse = np.zeros((ntimes,3))
    data = []
    #x = gendata21(errortype,n,p,beta)
    for i in range(ntimes):
        x,y,u = gendata(errortype,n,p,beta)
        data.append([x,y,u])
        sdhat[i,:],sdhat2[i,:],sdhat3[i,:],sdhat4[i,:],sdhat5[i,:],sdhat6[i,:],sdhat7[i,:],sdhat8[i,:],betakdreall[i,:],betakdreallstar[i,:],betakdrealllse[i,:],thetahat,hall =  sdestsemi(x,y,u,alpha,typek,typek2,tolk)
        print(i)
        print(time.strftime('%H.%M.%S',time.localtime(time.time())))
    return sdhat,sdhat2,sdhat3,sdhat4,sdhat5,sdhat6,sdhat7,sdhat8,betakdreall,betakdreallstar,betakdrealllse,data


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






