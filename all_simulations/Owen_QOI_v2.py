# Model with Feedback Reduced version

import math
import numpy as np

#import warnings, sys
import time

import scipy, scipy.sparse, scipy.optimize
#import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint, NonlinearConstraint, BFGS, SR1
from scipy import interpolate, sparse
#from numba import jit

def QOI(pars, *args):

    rescaled, hx\
    , Dh, Db, Di, Da\
    , ValH, ValB, ValI, ValA\
    , Kbind, Ncell, Nedge\
    , USol\
    , BicValR\
    , Hconc, Iconc, Bconc, Aconc, DPsi\
    , XcellExtend, XedgeExtend\
    , ThetaS, ThetaShx\
    , UpwindL, UpwindM\
    , data_main, indices, indptr\
    , startA, startB, startC, startH\
    , Ncellp2, Nedgep2, Nrows, Ncols\
    ,Nstart_Hop  ,Nstart_ConH \
    ,Nstart_DfutH ,Nstart_Aop  \
    ,Nstart_BonA ,Nstart_DfutA\
    ,Nstart_AonB ,Nstart_Bop  \
    ,Nstart_DfutB ,Nstart_HonC \
    ,Nstart_Cop  ,Nstart_DfutC\
    , Nstart_eye\
    , SolVelValL, SolValL\
    = args
    #, SolVeValL\
    #, HydExchangerParam\
    #, BicExchangerParam\


    start = time.time()

    #Now we need to change the parameters of the steady state problem. 
    #First lets do the concentration values at the right

    #Please do NOT change pars below

    #Hright
    #HVR
    HydValR = 10**pars[0]

    #Bright
    #BicValR = pars[1]

    #Cright
    #IVR
    IonValR = 10**pars[1]

    #KHI
    #HER
    HydExchangeRate = 10**pars[2]
    #HydExchangeRate = 1e-7
    

    #KBA
    #BER
    BicExchangeRate = 10**pars[3]

    #HEP
    HydExchangerParam = pars[4]

    #BEP
    BicExchangerParam = pars[5]

    #Electroneutrality
    #Bright
    AniValR = HydValR + IonValR - BicValR

    #Source mag
    #source_mag = 10**pars[6]
    source_mag = 1
    rescaled = source_mag * rescaled





    #Tolerance to determine when we stop iterating
    tolerance = 5e-10
    #tolerance = 5e-9
    #Initialize an error variable to large value
    maxer = 1

    j = 0

    #Maximum Iterations
    Iters = 100000

    #Initialization of Variables
    Hnow = np.copy(Hconc)
    Bnow = np.copy(Bconc)
    Cnow = np.copy(Iconc)
    Anow = np.copy(Aconc)
    Pnow = np.copy(DPsi)
    
    
    #Arrays to store my error measure
    Her = np.zeros((Iters,1))
    Ber = np.zeros((Iters,1))
    Cer = np.zeros((Iters,1))
    Aer = np.zeros((Iters,1))
    
    #ThetaShx = hx * ThetaS

    #This is the operator which acts on anions. It doesn't change depending
    #on concentrations!
    
    HydRHS = -rescaled
    AniRHS = np.copy(HydRHS)
    
    
    BicRHS = 0*rescaled
    CatRHS = 0*rescaled
    
    HydRHS_t = np.zeros((Ncellp2,))
    AniRHS_t = np.zeros((Ncellp2,))
    BicRHS_t = np.zeros((Ncellp2,))
    CatRHS_t = np.zeros((Ncellp2,))
    
    HydRHS_t[0] = 0
    HydRHS_t[1:-1] = HydRHS
    #HydRHS_t[-1] = HydValR
    
    AniRHS_t[0] = 0
    AniRHS_t[1:-1] = AniRHS
    #AniRHS_t[-1] = AniValR
    
    CatRHS_t[0] = 0
    CatRHS_t[1:-1] = CatRHS
    #CatRHS_t[-1] = IonValR
    
    BicRHS_t[0] = 0
    BicRHS_t[1:-1] = BicRHS
    #BicRHS_t[-1] = BicValR
    
    BigRHS = np.zeros((4*Ncellp2+len(Iconc)-1,))
    BigRHS[0:Ncellp2] = HydRHS_t 
    BigRHS[Ncellp2:2*Ncellp2] = AniRHS_t
    BigRHS[2*Ncellp2:3*Ncellp2] = BicRHS_t
    BigRHS[3*Ncellp2:4*Ncellp2] = CatRHS_t

    BigRHS[Ncellp2-1] = HydValR
    BigRHS[2*Ncellp2-1] = AniValR
    BigRHS[3*Ncellp2-1] = BicValR
    BigRHS[4*Ncellp2-1] = IonValR
    
    it = 0

    ##########################################################
    ##  Diffusion Operator Construct
    ##########################################################
    def DiffusionOperatorConstruct(DiffCoeff,BndFluxCoeff):
        
        #For ease of typing, we will make a locally scoped array for volume
        #fraction
        workingSol = ThetaS
        edgeSol = (ThetaS[0:-1] + ThetaS[1:])/2
        edgeSol[0] = SolValL
        
        #And the diffusion coefficient, whos name i don't want to type over & over
        D = DiffCoeff
        
        #Puting together the Laplacian-like operator for the Solvent
        LSolUDiag = D*np.divide(edgeSol[1:], hx**2 * workingSol[1:-1])
        LSolLDiag = D*np.divide(edgeSol[0:-1], hx**2 * workingSol[1:-1])
        LSolMDiag = -D*np.divide(edgeSol[0:-1] + edgeSol[1:], hx**2 * workingSol[1:-1]) 
    
        N = len(LSolMDiag)
        LImpUDiag = np.zeros((N+1,))
        LImpMDiag = np.zeros((N+2,)) 
        LImpLDiag = np.zeros((N+1,))
        
        LImpUDiag[0] = -D/hx - BndFluxCoeff/2
        LImpUDiag[1:] = LSolUDiag
    
        LImpMDiag[0] = SolVelValL + D/hx - BndFluxCoeff/2 
        LImpMDiag[1:-1] = LSolMDiag
        LImpMDiag[-1] = 1/2
    
        LImpLDiag[0:-1] = LSolLDiag
        LImpLDiag[-1] = 1/2
        
        #Final variable coefficient implicit operator
        #Ldiff = scipy.sparse.diags([LImpLDiag,LImpMDiag,LImpUDiag],[-1,0,1],shape=(Ncell+2,Ncell+2));
        #Ldiff = Ldiff.toarray()
    
        return LImpLDiag, LImpMDiag, LImpUDiag

    #Here are the diffusion operators. They include the linear part of the
    #left boundary condition
    LhydL,  LhydM, LhydU = DiffusionOperatorConstruct(Dh,-HydExchangeRate*HydExchangerParam)
    LaniL,  LaniM, LaniU = DiffusionOperatorConstruct(Da,-BicExchangeRate)
    LbicL,  LbicM, LbicU = DiffusionOperatorConstruct(Db,-BicExchangeRate*BicExchangerParam)
    LcatL,  LcatM, LcatU = DiffusionOperatorConstruct(Di,-HydExchangeRate)

    HopM = LhydM - UpwindM
    HopL = LhydL - UpwindL
    HopU = LhydU

    BopM = LbicM - UpwindM
    BopL = LbicL - UpwindL
    BopU = LbicU
    #print(BopL)
    #print(BopU)
    #input('w')

    AopM = LaniM - UpwindM
    AopL = LaniL - UpwindL
    AopU = LaniU

    CopM = LcatM - UpwindM
    CopL = LcatL - UpwindL
    CopU = LcatU

    # Putting in ConH
    data_main[Nstart_ConH] = -HydExchangeRate/2
    data_main[Nstart_ConH+1] = -HydExchangeRate/2
    
    #Putting in HonC
    data_main[Nstart_HonC] = -HydExchangeRate*HydExchangerParam/2
    data_main[Nstart_HonC+1] = -HydExchangeRate*HydExchangerParam/2

    # Putting in AonB
    data_main[Nstart_AonB] = -BicExchangeRate/2
    data_main[Nstart_AonB+1] = -BicExchangeRate/2

    # Putting in BonA
    data_main[Nstart_BonA] = -BicExchangeRate*BicExchangerParam/2
    data_main[Nstart_BonA+1] = -BicExchangeRate*BicExchangerParam/2


    #print(HopU)
    #input('w')

    # Putting in lower and upper diagonals of Hop, Bop, Aop, Cop
    data_main[Nstart_Hop+1] = HopU[0]
    data_main[Nstart_Bop+1] = BopU[0]
    data_main[Nstart_Aop+1] = AopU[0]
    data_main[Nstart_Cop+1] = CopU[0]

    for j in range(0, Ncellp2-2):
        #Upper Hop Diagonal
        data_main[Nstart_Hop+7 + 5*j] = HopU[j+1]
        #Lower Hop Diagonal
        data_main[Nstart_Hop+5 + 5*j] = HopL[j]

        #Upper Bop Diagonal
        data_main[Nstart_Bop+5 + 5*j] = BopU[j+1]
        #Lower Bop Diagonal
        data_main[Nstart_Bop+3 + 5*j] = BopL[j]


        data_main[Nstart_Aop+7 + 5*j] = AopU[j+1]
        data_main[Nstart_Aop+6 + 5*j] = AopM[j+1]
        data_main[Nstart_Aop+5 + 5*j] = AopL[j]

        data_main[Nstart_Cop+5 + 5*j] = CopU[j+1]
        data_main[Nstart_Cop+4 + 5*j] = CopM[j+1]
        data_main[Nstart_Cop+3 + 5*j] = CopL[j]


    #Last entry of lower diagonal of Hop
    data_main[Nstart_Aop-2] = HopL[-1]
    data_main[Nstart_HonC-2] = BopL[-1]
    data_main[Nstart_AonB-2] = AopL[-1]
    data_main[Nstart_eye-2] = CopL[-1]

    # Putting in first and last entries of main diagonal for HABCop
    data_main[Nstart_Hop] = HopM[0]
    data_main[Nstart_Aop-1] = HopM[-1]

    data_main[Nstart_Bop] = BopM[0]
    data_main[Nstart_HonC-1] = BopM[-1]

    data_main[Nstart_Aop] = AopM[0]
    data_main[Nstart_AonB-1] = AopM[-1]

    data_main[Nstart_Cop] = CopM[0]
    data_main[Nstart_eye-1] = CopM[-1]



    #Timing
    #start = time.time()

    while maxer > tolerance: 

        it = it + 1
        
        #Save the previous iteration of each variable
        Hold = np.copy(Hnow)
        Bold = np.copy(Bnow)
        Cold = np.copy(Cnow)
        Aold = np.copy(Anow)
        Pold = np.copy(Pnow)

        weightedges_H = ValH * Dh * np.interp(XedgeExtend, XcellExtend, ThetaS*Hnow)

        weightedges_A = ValA * Da * np.interp(XedgeExtend, XcellExtend, ThetaS*Anow)

        weightedges_B = ValB * Db * np.interp(XedgeExtend, XcellExtend, ThetaS*Bnow)

        weightedges_C = ValI * Di * np.interp(XedgeExtend, XcellExtend, ThetaS*Cnow)

        for j in range(1,Ncellp2-1):

            #Main Hop diagonal 1 to Ncellp2-1
            data_main[Nstart_Hop + 6 + 5*(j-1)] = HopM[j] - Kbind * Bnow[j]

            #Main Bop diagonal 1 to Ncellp2-1
            data_main[Nstart_Bop + 4 + 5*(j-1)] =  BopM[j] - Kbind * Hnow[j]
           
            #Divergence operators for potential gradient in hydrogen equations
            #Lower Diagnonal second element to end
            data_main[Nstart_DfutH+4 + 5*(j-1)] =  -weightedges_H[j-1]/ThetaShx[j]
            #Main Diagonal                 
            data_main[Nstart_DfutH+5 + 5*(j-1)] = weightedges_H[j]/ThetaShx[j]
                                          
            data_main[Nstart_DfutA+4 + 5*(j-1)] =  -weightedges_A[j-1]/ThetaShx[j]
            #Main Diagonal               
            data_main[Nstart_DfutA+5 + 5*(j-1)] = weightedges_A[j]/ThetaShx[j]
                                        
            data_main[Nstart_DfutB+4 + 5*(j-1)] =  -weightedges_B[j-1]/ThetaShx[j]
            #Main Diagonal             
            data_main[Nstart_DfutB+5 + 5*(j-1)] = weightedges_B[j]/ThetaShx[j]
                                      
            data_main[Nstart_DfutC+4 + 5*(j-1)] =  -weightedges_C[j-1]/ThetaShx[j]
            #Main Diagonal           
            data_main[Nstart_DfutC+5 + 5*(j-1)] = weightedges_C[j]/ThetaShx[j]



        data_main[Nstart_DfutH] = -ValH*Dh*(Hnow[0] + Hnow[1])/2
        data_main[Nstart_DfutA] = -ValA*Da*(Anow[0] + Anow[1])/2
        data_main[Nstart_DfutB] = -ValB*Db*(Bnow[0] + Bnow[1])/2
        data_main[Nstart_DfutC] = -ValI*Di*(Cnow[0] + Cnow[1])/2



        #plt.plot(BigRHS)
        #plt.show()
        #input('w')
        #print(len(data_main), len(indices))
        #input('w')

        L_sparse = scipy.sparse.csr_matrix((data_main, indices, indptr), shape=(Nrows, Nrows))
        #L_full = L_sparse.todense()
        #Bop_test = L_full[2*Ncellp2:3*Ncellp2, 2*Ncellp2:3*Ncellp2]
        #np.savetxt('Bop_test.csv', Bop_test, delimiter = ',')
        #input('w')
        #print(L_sparse[2*Ncellp2:3*Ncellp2, 2*Ncellp2:3*Ncellp2])
        #input('w')

        foo = scipy.sparse.linalg.spsolve(L_sparse, BigRHS, permc_spec='MMD_AT_PLUS_A')  

        #foo = scipy.linalg.solve(L_full, BigRHS)

        #print(finish - start)
    
        Hnow = foo[0:Ncellp2]
        Anow = foo[Ncellp2:2*Ncellp2]
        Bnow = foo[2*Ncellp2:3*Ncellp2]
        Cnow = foo[3*Ncellp2:4*Ncellp2]
        Pnow = foo[4*Ncellp2:]


        Her[j-1] = np.max(np.abs(Hnow-Hold))
        Ber[j-1] = np.max(np.abs(Bnow-Bold))
        Aer[j-1] = np.max(np.abs(Anow-Aold))
        Cer[j-1] = np.max(np.abs(Cnow-Cold))

        maxer = np.amax([Her[j-1],Aer[j-1],Ber[j-1],Cer[j-1]]);
        #print(it, maxer, end='\r')
        #if it%100 == 0:
        #    print(it, maxer, end='\n')
        #finish = time.time()
        #if finish -  start > 30*60:
        #    print(pars)

    #print(Bnow)

    #finish = time.time()
    #print('iterations: ', it, ' time taken: ', finish - start)
    val = (Hnow[0] + Hnow[1])/2
    if val < 0:
        #if val < -1e-14:
        log_val = np.log10(-(Hnow[0] + Hnow[1])/2)
    else:
        log_val = np.log10((Hnow[0] + Hnow[1])/2)
        
    
    #return np.array([val, log_val])
    return np.array([log_val])
    #return Hnow, Anow, Bnow, Cnow, Pnow


    
