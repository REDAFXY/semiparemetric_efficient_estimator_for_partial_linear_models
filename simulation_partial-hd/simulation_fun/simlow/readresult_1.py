import numpy as np


'''------------------------------------------------------------------------------------------------------'''
'''4. main---simulation-table1
---4.1 table6-prederror
    ---generatedata
    ---predsim
    ---calestimate
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


def calestimate(ntimes,resultall,n):
    '''计算统计量的函数'''
    p=3
    beta_true=np.array([3,1.5,2]).reshape(p,1)
    #betaall,parmus,nonpamus,allmus,dataall = resultall
    betaall,parmus,nonpamus,allmus,hallre,dataall = resultall
    predTP = np.zeros((ntimes,3))
    prednonpaerr=np.zeros((ntimes,3))
    predFP=np.zeros((ntimes,3))
    predpaerr=np.zeros((ntimes,3))
    predmse = np.zeros((ntimes,3))
    predallerr=np.zeros((ntimes,3))
    predmad=np.zeros((ntimes,3))
    for i in range(ntimes):
        for j in range(3):
            beta = betaall[i,:,j].reshape(p,1)
            predmse[i,j] = np.sum((beta-beta_true)**2)
            predTP[i,j],predFP[i,j],predmad[i,j] = (beta-beta_true)**2
            predallerr[i,j] = np.mean((allmus[i,:,j])**2)*100
            prednonpaerr[i,j] = np.mean((nonpamus[i,:,j])**2)*100
            predpaerr[i,j] = np.mean((parmus[i,:,j])**2)*100
            #predmad[i,j] = np.sum(np.abs(beta-beta_true))    
    predmse=np.hstack((np.mean(predmse,axis=0),np.std(predmse,axis=0)))
    predTP=np.hstack((np.mean(predTP,axis=0),np.std(predTP,axis=0)))
    predpaerr=np.hstack((np.mean(predpaerr,axis=0),np.std(predpaerr,axis=0)))
    predFP=np.hstack((np.mean(predFP,axis=0),np.std(predFP,axis=0)))
    prednonpaerr=np.hstack((np.mean(prednonpaerr,axis=0),np.std(prednonpaerr,axis=0)))
    predallerr=np.hstack((np.mean(predallerr,axis=0),np.std(predallerr,axis=0)))
    predmad=np.hstack((np.mean(predmad,axis=0),np.std(predmad,axis=0)))
    bic = np.hstack((predmse,predTP,predpaerr,predFP,prednonpaerr,predallerr,predmad))
    return bic

def calestimatemap(ntimes,resultall,n):
    '''计算统计量的函数'''
    p=3
    beta_true=np.array([3,1.5,2]).reshape(p,1)
    #betaall,parmus,nonpamus,allmus,dataall = resultall
    betaall,parmus,nonpamus,allmus,hallre,dataall = resultall
    predTP = np.zeros((ntimes,3))
    prednonpaerr=np.zeros((ntimes,3))
    predFP=np.zeros((ntimes,3))
    predpaerr=np.zeros((ntimes,3))
    predmse = np.zeros((ntimes,3))
    predallerr=np.zeros((ntimes,3))
    predmad=np.zeros((ntimes,3))
    for i in range(ntimes):
        for j in range(3):
            beta = betaall[i,:,j].reshape(p,1)
            beta = betaall[i,:,j].reshape(p,1)
            predmse[i,j] = np.sum((beta-beta_true)**2)
            predTP[i,j],predFP[i,j],predmad[i,j] = (beta-beta_true)**2
            predallerr[i,j] = np.median(np.abs(allmus[i,:,j]))*100
            prednonpaerr[i,j] = np.median(np.abs(nonpamus[i,:,j]))*100
            predpaerr[i,j] = np.median(np.abs(parmus[i,:,j]))*100 
    predmse=np.hstack((np.median(predmse,axis=0),np.std(predmse,axis=0)))
    predTP=np.hstack((np.median(predTP,axis=0),np.std(predTP,axis=0)))
    predpaerr=np.hstack((np.median(predpaerr,axis=0),np.std(predpaerr,axis=0)))
    predFP=np.hstack((np.median(predFP,axis=0),np.std(predFP,axis=0)))
    prednonpaerr=np.hstack((np.median(prednonpaerr,axis=0),np.std(prednonpaerr,axis=0)))
    predallerr=np.hstack((np.median(predallerr,axis=0),np.std(predallerr,axis=0)))
    predmad=np.hstack((np.median(predmad,axis=0),np.std(predmad,axis=0)))
    bic = np.hstack((predmse,predTP,predpaerr,predFP,prednonpaerr,predallerr,predmad))
    return bic



'''------------------------------------------------------------------------------------------------------'''
'''---4.2 table1-mse/mad:
    ---SIMbeta：估计beta
    ---writemse：存mse结果
    ---writemad：存mad结果
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''

def writemse(ntimes,betaresultall):
    betalse,betalsekdre,betalekdre = betaresultall[:,:,0],betaresultall[:,:,1],betaresultall[:,:,2]
    p=3
    beta=np.array([3,1.5,2])
    mse_es0=[]
    mse_es2=[]
    mse_es4=[]
    for i in range(ntimes):
        mse_es0.append((beta.reshape(p,1)-betalse[i,:].reshape(p,1))**2)
        mse_es2.append((beta.reshape(p,1)-betalsekdre[i,:].reshape(p,1))**2)
        mse_es4.append((beta.reshape(p,1)-betalekdre[i,:].reshape(p,1))**2)
    return np.mean(mse_es0,axis=0),np.std(mse_es0,axis=0),np.mean(mse_es2,axis=0),np.std(mse_es2,axis=0),np.mean(mse_es4,axis=0),np.std(mse_es4,axis=0)

 
def writemad(ntimes,betaresultall):
    betalse,betalsekdre,betalekdre = betaresultall[:,:,0],betaresultall[:,:,1],betaresultall[:,:,2]
    p=3
    beta=np.array([3,1.5,2])
    mse_es0=[]
    mse_es2=[]
    mse_es4=[]
    for i in range(ntimes):
        mse_es0.append(np.abs(beta.reshape(p,1)-betalse[i,:].reshape(p,1)))
        mse_es2.append(np.abs(beta.reshape(p,1)-betalsekdre[i,:].reshape(p,1)))
        mse_es4.append(np.abs(beta.reshape(p,1)-betalekdre[i,:].reshape(p,1)))
    return np.mean(mse_es0,axis=0),np.std(mse_es0,axis=0),np.mean(mse_es2,axis=0),np.std(mse_es2,axis=0),np.mean(mse_es4,axis=0),np.std(mse_es4,axis=0)

'''------------------------------------------------------------------------------------------------------'''
'''---4.3 table2-SD
    ---KDRESIM:SEMI*-SD
    ---KDRESIM2：SEMI-SD
------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------'''


def KDRESIM(ntimes,betaall,sdall):
    #lsekdre
    p = 3
    aaa,bbb,ccc = sdall[-1],sdall[-2],sdall[1]
    betakdre = betaall[:,:,1]
    sd_es=[]
    sd_es2=[]
    sd_es3=[]
    mean_es=[]
    for i in range(ntimes):
        mean_es.append(betakdre[i,:].reshape(p,1))
        sd_es.append(aaa[i,:].reshape(p,1))
        sd_es2.append(bbb[i,:].reshape(p,1))
        sd_es3.append(ccc[i,:].reshape(p,1))
    return np.mean(mean_es,axis=0),np.std(mean_es,axis=0),np.mean(sd_es,axis=0),np.std(sd_es,axis=0),np.mean(sd_es2,axis=0),np.std(sd_es2,axis=0),np.mean(sd_es3,axis=0),np.std(sd_es3,axis=0)



def KDRESIM2(ntimes,betaall,sdall):
    #kdre
    p = 3
    aaa,bbb,ccc = sdall[-1],sdall[-2],sdall[1]
    betakdre = betaall[:,:,2]
    sd_es=[]
    sd_es2=[]
    sd_es3=[]
    mean_es=[]
    for i in range(ntimes):
        mean_es.append(betakdre[i,:].reshape(p,1))
        sd_es.append(aaa[i,:].reshape(p,1))
        sd_es2.append(bbb[i,:].reshape(p,1))
        sd_es3.append(ccc[i,:].reshape(p,1))
    return np.mean(mean_es,axis=0),np.std(mean_es,axis=0),np.mean(sd_es,axis=0),np.std(sd_es,axis=0),np.mean(sd_es2,axis=0),np.std(sd_es2,axis=0),np.mean(sd_es3,axis=0),np.std(sd_es3,axis=0)




