import numpy as np

def calestimate(ntimes,resultall):
    '''计算统计量的函数'''
    p=1000
    beta_true=np.zeros(p).reshape(p,1)
    beta_true[0:5,0]=np.array([3,1.5,0,0,2])
    #betaall,parmus,nonpamus,allmus,dataall = resultall
    betaall,parmus,nonpamus,allmus,ldssre,hallre = resultall
    predTP = np.zeros((ntimes,6))
    prednonpaerr=np.zeros((ntimes,6))
    predFP=np.zeros((ntimes,6))
    predpaerr=np.zeros((ntimes,6))
    predmse = np.zeros((ntimes,6))
    predallerr=np.zeros((ntimes,6))
    predmad=np.zeros((ntimes,6))
    for i in range(ntimes):
        for j in range(6):
            beta = betaall[i,:,j].reshape(p,1)
            predmse[i,j] = np.sum((beta-beta_true)**2)
            predTP[i,j] = np.sum((beta_true!=0)*(beta!=0))
            predFP[i,j] = np.sum((beta_true==0)*(beta!=0))
            #这里是sum(_1^50)*2所以等于mean(_1^50)*100
            predallerr[i,j] = np.sum((allmus[i,:,j])**2)*2
            prednonpaerr[i,j] = np.sum((nonpamus[i,:,j])**2)*2
            predpaerr[i,j] = np.sum((parmus[i,:,j])**2)*2
            predmad[i,j] = np.sum(np.abs(beta-beta_true))    
    predmse=np.hstack((np.mean(predmse,axis=0),np.std(predmse,axis=0)))
    predTP=np.hstack((np.mean(predTP,axis=0),np.std(predTP,axis=0)))
    predpaerr=np.hstack((np.mean(predpaerr,axis=0),np.std(predpaerr,axis=0)))
    predFP=np.hstack((np.mean(predFP,axis=0),np.std(predFP,axis=0)))
    prednonpaerr=np.hstack((np.mean(prednonpaerr,axis=0),np.std(prednonpaerr,axis=0)))
    predallerr=np.hstack((np.mean(predallerr,axis=0),np.std(predallerr,axis=0)))
    # mades = np.median(predmad,axis=0)
    # mapes = np.abs(predmad - mades)
    # predmad=np.hstack((mades,np.median(mapes,axis=0)))
    predmad=np.hstack((np.mean(predmad,axis=0),np.std(predmad,axis=0)))
    bic = np.hstack((predmse,predTP,predpaerr,predFP,prednonpaerr,predallerr,predmad))
    return bic

def calestimatemap(ntimes,resultall):
    '''计算统计量的函数'''
    p=1000
    beta_true=np.zeros(p).reshape(p,1)
    beta_true[0:5,0]=np.array([3,1.5,0,0,2])
    #betaall,parmus,nonpamus,allmus,dataall = resultall
    betaall,parmus,nonpamus,allmus,ldssre,hallre = resultall
    predTP = np.zeros((ntimes,6))
    prednonpaerr=np.zeros((ntimes,6))
    predFP=np.zeros((ntimes,6))
    predpaerr=np.zeros((ntimes,6))
    predmse = np.zeros((ntimes,6))
    predallerr=np.zeros((ntimes,6))
    predmad=np.zeros((ntimes,6))
    for i in range(ntimes):
        for j in range(6):
            beta = betaall[i,:,j].reshape(p,1)
            predmse[i,j] = np.sum((beta-beta_true)**2)
            predTP[i,j] = np.sum((beta_true!=0)*(beta!=0))
            predFP[i,j] = np.sum((beta_true==0)*(beta!=0))
            #这里是sum(_1^50)*2所以等于mean(_1^50)*100
            predallerr[i,j] = np.sum(np.abs(allmus[i,:,j]))*2
            prednonpaerr[i,j] = np.sum(np.abs(nonpamus[i,:,j]))*2
            predpaerr[i,j] = np.sum(np.abs(parmus[i,:,j]))*2
            predmad[i,j] = np.sum(np.abs(beta-beta_true))    
    predmse=np.hstack((np.mean(predmse,axis=0),np.std(predmse,axis=0)))
    predTP=np.hstack((np.mean(predTP,axis=0),np.std(predTP,axis=0)))
    predpaerr=np.hstack((np.mean(predpaerr,axis=0),np.std(predpaerr,axis=0)))
    predFP=np.hstack((np.mean(predFP,axis=0),np.std(predFP,axis=0)))
    prednonpaerr=np.hstack((np.mean(prednonpaerr,axis=0),np.std(prednonpaerr,axis=0)))
    predallerr=np.hstack((np.mean(predallerr,axis=0),np.std(predallerr,axis=0)))
    # mades = np.median(predmad,axis=0)
    # mapes = np.abs(predmad - mades)
    # predmad=np.hstack((mades,np.median(mapes,axis=0)))
    predmad=np.hstack((np.mean(predmad,axis=0),np.std(predmad,axis=0)))
    bic = np.hstack((predmse,predTP,predpaerr,predFP,prednonpaerr,predallerr,predmad))
    return bic




