import itertools
import numpy as np
import pandas as pd
from progress.bar import Bar


def MahalanobisDistance(YObs,EpsObs,Thetas,YTheo,YOther):
    #-> Original Author: May Gade Pedersen
    #-> Updated: Cole Johnston 2018-06-27

    # N: number of observed values
    # p: number of varied parameters in the grid (e.g. Mini, Xini, Xc etc.)
    # q: number of grid points
    # YObs: observed values, e.g. frequencies (a vector of length N)
    # EpsObs: errors on the observed values (a vector of length N)
    # Thetas: Matrix of parameters in the grid (dimensions q x p)
    # YTheo: corresponding theoretical values of the observed ones
    # (a vector of length N)

    # Determine number of grid points
    q = np.shape(Thetas)[0]
    # print("N grid points --> ",q)

    # Convert to matrix format
    YObsMat = np.matrix(YObs).T
    YTheoMat = np.matrix(YTheo).T
    YOtherMat = np.matrix(YOther).T

    # Calculate the average on the theoretical values (e.g. frequencies)
    # over the entire grid. Returns a vector of dimension N x 1
    Yav = YTheoMat.mean(1)
    # print("Average Values -->")
    # print(Yav)

    # Calculate the variance-covriance matrix
    N = len(YObs)
    V = np.zeros((N,N))

    vBar = Bar('Calculating Correlation Matrix-->',max=q)
    for i in range(q):
        difference = np.subtract(YTheoMat[:,i],Yav)
        V += np.matmul(difference,difference.T)
        vBar.next()
    vBar.finish()
    V = V/(q-1.)

    # Include observational errors in the variance-covariance matrix
    V = V + np.diag(EpsObs**2.)


    # Calculate Mahalanobis distances
    MD = np.zeros(q)
    Vinv = np.linalg.pinv(V,rcond=1e-12)
    mdBAR = Bar('Running MD -->',max=q)
    for i in range(q):
        diff = (YTheoMat[:,i]-YObsMat)
        MD[i] = np.matmul(np.matmul(diff.T,Vinv),diff)[0][0]
        mdBAR.next()
    mdBAR.finish()
    # return the results ordered from lowest to highest Mahalanobis
    # distance
    idx2 = np.argsort(MD)
    return np.concatenate((np.matrix(MD[idx2]).T,Thetas[idx2,:],YTheoMat[:,idx2].T,YOtherMat[:,idx2].T),axis=1)


def MahalanobisDistanceTEST(YObs,EpsObs,Thetas,YTheo,YOther):
    #-> Original Author: May Gade Pedersen
    #-> Updated: Cole Johnston 2018-06-27

    # N: number of observed values
    # p: number of varied parameters in the grid (e.g. Mini, Xini, Xc etc.)
    # q: number of grid points
    # YObs: observed values, e.g. frequencies (a vector of length N)
    # EpsObs: errors on the observed values (a vector of length N)
    # Thetas: Matrix of parameters in the grid (dimensions q x p)
    # YTheo: corresponding theoretical values of the observed ones
    # (a vector of length N)

    # Determine number of grid points
    q = np.shape(Thetas)[0]
    # print("N grid points --> ",q)

    # Convert to matrix format
    YObsMat = np.matrix(YObs).T
    YTheoMat = np.matrix(YTheo).T
    YOtherMat = np.matrix(YOther).T

    # Calculate the average on the theoretical values (e.g. frequencies)
    # over the entire grid. Returns a vector of dimension N x 1
    Yav = YTheoMat.mean(1)
    # print("Average Values -->")
    # print(Yav)

    # Calculate the variance-covriance matrix
    N = len(YObs)
    V = np.zeros((N,N))

    vBar = Bar('Calculating Correlation Matrix-->',max=q)
    for i in range(q):
        difference = np.subtract(YTheoMat[:,i],Yav)
        V += np.matmul(difference,difference.T)
        vBar.next()
    vBar.finish()
    V = V/(q-1.)

    # Include observational errors in the variance-covariance matrix
    V = V + np.diag(EpsObs**2.)
    # V = V.T

    # Calculate Mahalanobis distances
    MD = np.zeros(q)
    Vinv = np.linalg.pinv(V,rcond=1e-12)
    # mdBAR = Bar('Running MD -->',max=q)
    diff = np.subtract(YTheoMat,YObsMat)

    print(np.shape(diff))
    print(np.shape(Vinv))
    print(np.shape(diff.T))

    MD = np.einsum('nj,jk,kn->n', diff.T, Vinv, diff)
    # return the results ordered from lowest to highest Mahalanobis
    # distance
    print(np.shape(MD))
    idx2 = np.argsort(MD)

    print(np.shape(Thetas),np.shape(Thetas.T))
    print(np.shape(YTheoMat),np.shape(YTheoMat.T))
    print(np.shape(YOtherMat),np.shape(YOtherMat.T))
    return np.concatenate((np.matrix(MD[idx2]).T,Thetas[idx2,:],YTheoMat[:,idx2].T,YOtherMat[:,idx2].T),axis=1)


def MonteCarlo_MahalanobisDistance(YObs,EpsObs,Thetas,YTheo,YOther,niters,MD_tags):
    #-> Original Author: Cole Johnston 2019-01-10 ; YMD

    # N: number of observed values
    # p: number of varied parameters in the grid (e.g. Mini, Xini, Xc etc.)
    # q: number of grid points
    # YObs: observed values, e.g. frequencies (a vector of length N)
    # EpsObs: errors on the observed values (a vector of length N)
    # Thetas: Matrix of parameters in the grid (dimensions q x p)
    # YTheo: corresponding theoretical values of the observed ones
    # (a vector of length N)

    # Determine number of grid points
    q = np.shape(Thetas)[0]

    # -> Here, we need to make an array of perturbed YObs values
    # create niters number of matrices randomly
    # perturbed about the observed values
    shape_MC = (niters,len(YObs))
    YObs_MC = np.random.normal(YObs,EpsObs,shape_MC)

    # Convert to matrix format
    # YObsMat = np.matrix(YObs).T
    YTheoMat = np.matrix(YTheo).T
    YOtherMat = np.matrix(YOther).T

    # Calculate the average on the theoretical values (e.g. frequencies)
    # over the entire grid. Returns a vector of dimension N x 1
    Yav = YTheoMat.mean(1)


    # Calculate the variance-covriance matrix
    N = len(YObs)
    V = np.zeros((N,N))

    vBar = Bar('Calculating Correlation Matrix-->',max=q)
    for i in range(q):
        difference = np.subtract(YTheoMat[:,i],Yav)
        V += np.matmul(difference,difference.T)
        vBar.next()
    vBar.finish()
    V = V/(q-1.)

    # Include observational errors in the variance-covariance matrix
    V = V + np.diag(EpsObs**2.)
    Vinv = np.linalg.pinv(V,rcond=1e-12)


    # Calculate Mahalanobis distances for each iteration
    # each entry in YObs_MC is a perturbed YObs array
    MC_MD_output_df = pd.DataFrame({tag:[] for tag in MD_tags})
    mcBAR = Bar('Running MC -->',max=niters)
    for jj in range(niters):
        YObsMat = np.matrix(YObs_MC[jj]).T
        diff = np.subtract(YTheoMat,YObsMat)
        MD = np.einsum('nj,jk,kn->n', diff.T, Vinv, diff)
        idx = np.argsort(MD)
        MD_output = np.concatenate((np.matrix(MD[idx]).T,Thetas[idx,:],YTheoMat[:,idx].T,YOtherMat[:,idx].T),axis=1)
        MD_output_df = pd.DataFrame({tag: np.ravel(MD_output[:,ii],order='A')[:100] for ii,tag in enumerate(MD_tags)})
        MC_MD_output_df = MC_MD_output_df.append(MD_output_df,ignore_index=True)
        mcBAR.next()
    mcBAR.finish()
    # return the results ordered from lowest to highest Mahalanobis
    # distance

    MC_MD_output_df = MC_MD_output_df.sort_values('MD')
    yield MC_MD_output_df


def MonteCarlo_MahalanobisDistance_Hugues(YObs,EpsObs,Thetas,YTheo,YOther,niters,MD_tags):
    #-> Original Author: Cole Johnston 2019-01-10 ; YMD

    # N: number of observed values
    # p: number of varied parameters in the grid (e.g. Mini, Xini, Xc etc.)
    # q: number of grid points
    # YObs: observed values, e.g. frequencies (a vector of length N)
    # EpsObs: errors on the observed values (a vector of length N)
    # Thetas: Matrix of parameters in the grid (dimensions q x p)
    # YTheo: corresponding theoretical values of the observed ones
    # (a vector of length N)

    # Determine number of grid points
    q = np.shape(Thetas)[0]

    # -> Here, we need to make an array of perturbed YObs values
    # create niters number of matrices randomly
    # perturbed about the observed values

    # Convert to matrix format
    YObsMat = np.matrix(YObs).T
    YTheoMat = np.matrix(YTheo).T
    YOtherMat = np.matrix(YOther).T

    # Calculate the average on the theoretical values (e.g. frequencies)
    # over the entire grid. Returns a vector of dimension N x 1
    Yav = YTheoMat.mean(1)


    # Calculate the variance-covriance matrix
    N = len(YObs)
    V = np.zeros((N,N))

    vBar = Bar('Calculating Correlation Matrix-->',max=q)
    for i in range(q):
        difference = np.subtract(YTheoMat[:,i],Yav)
        V += np.matmul(difference,difference.T)
        vBar.next()
    vBar.finish()
    V = V/(q-1.)

    # Include observational errors in the variance-covariance matrix
    Vinv_MC = np.linalg.pinv(V,rcond=1e-12)
    V = V + np.diag(EpsObs**2.)
    Vinv = np.linalg.pinv(V,rcond=1e-12)

    diff_best = np.subtract(YTheoMat,YObsMat)
    MD_best = np.einsum('nj,jk,kn->n', diff_best.T, Vinv, diff_best)
    idx_best = np.argsort(MD_best)
    Y_best =  YTheoMat[:,idx_best].T[0][0]
    shape_MC = (niters,len(YObs))
    YObs_MC = np.random.normal(Y_best,EpsObs,shape_MC)

    # Calculate Mahalanobis distances for each iteration
    # each entry in YObs_MC is a perturbed YObs array
    MC_MD_output_df = pd.DataFrame({tag:[] for tag in MD_tags})
    mcBAR = Bar('Running MC -->',max=niters)
    for jj in range(niters):
        YObsMat = np.matrix(YObs_MC[jj]).T
        diff = np.subtract(YTheoMat,YObsMat)
        MD = np.einsum('nj,jk,kn->n', diff.T, Vinv, diff)
        idx = np.argsort(MD)
        MD_output = np.concatenate((np.matrix(MD[idx]).T,Thetas[idx,:],YTheoMat[:,idx].T,YOtherMat[:,idx].T),axis=1)
        MD_output_df = pd.DataFrame({tag: np.ravel(MD_output[:,ii],order='A')[:1] for ii,tag in enumerate(MD_tags)})
        MC_MD_output_df = MC_MD_output_df.append(MD_output_df,ignore_index=True)
        mcBAR.next()
    mcBAR.finish()
    # return the results ordered from lowest to highest Mahalanobis
    # distance

    MC_MD_output_df = MC_MD_output_df.sort_values('MD')
    yield MC_MD_output_df



def MahalanobisDistanceEB(YObsArr,EpsObsArr,Thetas,YTheo,YOther):
    #-> Original Author: May Gade Pedersen
    #-> Updated: Cole Johnston 2018-06-27

    # N: number of observed values
    # p: number of varied parameters in the grid (e.g. Mini, Xini, Xc etc.)
    # q: number of grid points
    # YObsArr: array of observed values per star, e.g. frequencies (a vector of length N)
    # EpsObsArr: array of errors on the observed values per star (a vector of length N)
    # Thetas: Matrix of parameters in the grid (dimensions q x p)
    # YTheo: corresponding theoretical values of the observed ones
    # (a vector of length N)

    # Determine number of grid points
    q = np.shape(Thetas)[0]

    YObsMatArr = [ np.matrix(YObs).T for YObs in YObsArr ]
    # Convert to matrix format
    YObsMat = np.matrix(YObs).T
    YTheoMat = np.matrix(YTheo).T
    YOtherMat = np.matrix(YOther).T

    # Calculate the average on the theoretical values (e.g. frequencies)
    # over the entire grid. Returns a vector of dimension N x 1
    Yav = YTheoMat.mean(1)

    # Calculate the variance-covriance matrix
    N = len(YObs)
    V = np.zeros((N,N))

    for i in range(q):
        difference = np.subtract(YTheoMat[:,i],Yav)
        V += np.matmul(difference,difference.T)
    V = V/(q-1.)


    # Include observational errors in the variance-covariance matrix
    VInvArr = [ np.linalg.pinv(V + np.diag(EpsObs**2.)) for EpsObs in EpsObsArr ]

    # Calculate Mahalanobis distances
    MDArr = [ np.zeros(q) for xx in YObsArr ]
    MDTot = np.zeros(q)
    # Vinv = np.linalg.pinv(V,rcond=1e-12)
    for i in range(q):
        diffArr = [ (YTheoMat[:,i]-YObsMat) for YObsMat in YObsMatArr ]
        for jj,MD in enumerate(MDArr):
            MD[i] = np.matmul(np.matmul(diffArr[jj].T,VInvArr[jj]),diffArr[jj])[0][0]



    # return the results ordered from lowest to highest Mahalanobis
    # distance
    idx2 = np.argsort(MD)

    return np.concatenate((np.matrix(MD[idx2]).T,Thetas[idx2,:],YTheoMat[:,idx2].T,YOtherMat[:,idx2].T),axis=1)
