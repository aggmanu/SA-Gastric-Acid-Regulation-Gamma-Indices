import numpy as np
import math
import time
import sys

import scipy.io as spio
import scipy

from Owen_QOI_v2 import QOI
from joblib import Parallel, delayed
#from SolnCheck import SolnCheck
#import matplotlib.pyplot as plt

mat = spio.loadmat('InitialData.mat', struct_as_record= False,squeeze_me= True)
mat = spio.loadmat('InitialData.mat', struct_as_record= False,squeeze_me= True)

GelSimParams = mat['GelSimParams'].__dict__
GelState = mat['GelState'].__dict__
rescaled = mat['rescaled']

rescaled_temp = np.zeros((len(rescaled),))
rescaled_temp = rescaled
rescaled = rescaled_temp

hx = GelSimParams['hx']
Ncell = GelSimParams['Ncell']
Nedge = GelSimParams['Nedges']
ThetaS = GelState['ThetaS']
USol = GelState['USol']
SolValL = GelSimParams['SolValL']

Dh = GelSimParams['Dh'] 
Db = GelSimParams['Db']
Di = GelSimParams['Di']
Da = GelSimParams['Da']

ValH = GelSimParams['ValH'] 
ValB = GelSimParams['ValB']
ValI = GelSimParams['ValI']
ValA = GelSimParams['ValA']

Kbind = GelSimParams['Kbind']
Ncell = GelSimParams['Ncell']
Nedge = GelSimParams['Nedges']
#print('ncell', Ncell, 'nedge', Nedge)
#input('w')

#Check GelState or GelSimParams

HydExchangerParam = GelSimParams['HydExchangerParam']

BicExchangerParam = GelSimParams['BicExchangerParam']

Hconc = GelState['Hconc']
Iconc = GelState['Iconc']
Bconc = GelState['Bconc']
Aconc = GelState['Aconc']
DPsi = GelState['DPsi']

HydValR = GelSimParams['HydValR']
AniValR = GelSimParams['AniValR']
IonValR = GelSimParams['IonValR']
BicValR = GelSimParams['BicValR']
HydExchangeRate = GelSimParams['HydExchangeRate'] 
BicExchangeRate = GelSimParams['BicExchangeRate']

XcellExtend = GelState['XcellExtend']
XedgeExtend = GelState['XedgeExtend']

SolVelValL = GelSimParams['SolVelValL']

ThetaShx = hx * ThetaS


Ncellp2 = Ncell + 2
Nedgep2 = Nedge + 2

##########################################################
##  Define the data, indices, and indptr to construct the
##  sparse L matrix
##########################################################



# Number of elements in each of Hop, Bop, Cop, Aop
N_op = Ncellp2 + 2*(Ncellp2 - 1)

# Elements in each of Dfut
N_dfut = 2*Nedgep2 - 1

# Elements in each of ConH, HonC, BonA, AonB
N_ionion = 2

#  Elements in Eye
N_eye = Nedgep2

# Elements in each of the first four row-blocks
N_rowblock = N_op + N_ionion + N_dfut
#print(N_rowblock)
#input('w')


# Starting index of individual matrices
Nstart_Hop = 0
Nstart_ConH = Nstart_Hop + 2
Nstart_DfutH = Nstart_ConH + 2

Nstart_Aop = Nstart_Hop + N_rowblock
Nstart_BonA = Nstart_Aop + 2
Nstart_DfutA = Nstart_BonA + 2

Nstart_AonB = Nstart_Aop + N_rowblock
Nstart_Bop = Nstart_AonB + 2
Nstart_DfutB = Nstart_Bop + 2

Nstart_HonC = Nstart_AonB + N_rowblock
Nstart_Cop = Nstart_HonC + 2
Nstart_DfutC = Nstart_Cop + 2

Nstart_eyeH = Nstart_HonC + N_rowblock
Nstart_eyeA = Nstart_eyeH + 1
Nstart_eyeB = Nstart_eyeA + 1
Nstart_eyeC = Nstart_eyeB + 1


# Number of non-zero elements
#print(24*Nedgep2 + 4)
#input('w')
NNZ = 24*Nedgep2 + 8
data_main = np.zeros((NNZ,))

#   Create the cumulative row
# size of the 2D L matrix
Nrows = 4*Ncellp2 + Nedgep2
Ncols = Nrows
indptr = 5*np.ones((Nrows+1,), dtype=int)
indptr[0] = 0
indptr[Ncellp2] = 2
indptr[2*Ncellp2] = 2
indptr[3*Ncellp2] = 2
indptr[4*Ncellp2] = 2
indptr[4*Ncellp2+1:] = 4
indptr = np.cumsum(indptr)

# Create to store column indices

indices = []

# Adding col indices for the first row-block
indices.append(0)
indices.append(1)
indices.append(3*Ncellp2)
indices.append(3*Ncellp2+1)
indices.append(4*Ncellp2)

for j in range(0, Ncellp2-2):
    indices.append(0 + j)
    indices.append(1 + j)
    indices.append(2 + j)
    indices.append(4*Ncellp2 + j)
    indices.append(4*Ncellp2 + 1 + j)

indices.append(Ncellp2 - 2)
indices.append(Ncellp2 - 1)


# Adding col indices for the second row-block
indices.append(Ncellp2)
indices.append(Ncellp2 + 1)
indices.append(2*Ncellp2)
indices.append(2*Ncellp2+1)
indices.append(4*Ncellp2)

for j in range(0, Ncellp2-2):
    indices.append(Ncellp2 + 0 + j)
    indices.append(Ncellp2 + 1 + j)
    indices.append(Ncellp2 + 2 + j)
    indices.append(4*Ncellp2 + j)
    indices.append(4*Ncellp2 + 1 + j)

indices.append(2*Ncellp2 - 2)
indices.append(2*Ncellp2 - 1)


# Adding col indices for the third row-block
indices.append(Ncellp2)
indices.append(Ncellp2 + 1)
indices.append(2*Ncellp2)
indices.append(2*Ncellp2+1)
indices.append(4*Ncellp2)

for j in range(0, Ncellp2-2):
    indices.append(2*Ncellp2 + 0 + j)
    indices.append(2*Ncellp2 + 1 + j)
    indices.append(2*Ncellp2 + 2 + j)
    indices.append(4*Ncellp2 + j)
    indices.append(4*Ncellp2 + 1 + j)

indices.append(3*Ncellp2 - 2)
indices.append(3*Ncellp2 - 1)


# Adding col indices for the fourth row-block
indices.append(0)
indices.append(1)
indices.append(3*Ncellp2)
indices.append(3*Ncellp2+1)
indices.append(4*Ncellp2)

for j in range(0, Ncellp2-2):
    indices.append(3*Ncellp2 + 0 + j)
    indices.append(3*Ncellp2 + 1 + j)
    indices.append(3*Ncellp2 + 2 + j)
    indices.append(4*Ncellp2 + j)
    indices.append(4*Ncellp2 + 1 + j)

indices.append(4*Ncellp2 - 2)
indices.append(4*Ncellp2 - 1)


# Adding col indices for the fifth row-block (Eyes)

for j in range(0, Nedgep2):
    indices.append(0 + j)
    indices.append(Ncellp2 + j)
    indices.append(2*Ncellp2 + j)
    indices.append(3*Ncellp2 + j)

#print(NNZ, len(indices))
#input('w')

##########################################################
##  Put in Eye matrices in data_main
##########################################################

#   Eyes are after the four row-blocks
Nstart_eye = 4*N_rowblock

for j in range(Nedgep2):
    #EyeH
    data_main[Nstart_eye+ 4*j] = ValH
                   
    #EyeA         
    data_main[Nstart_eye+1 + 4*j] = ValA
                 
    #EyeB       
    data_main[Nstart_eye+2 + 4*j] = ValB
              
    #EyeC      
    data_main[Nstart_eye +3+ 4*j] = ValI



##########################################################
##  Upwind Operator Construct
##########################################################
def UpwindOperatorConstruct():

    #Puting together the Laplacian-like operator for the Solvent
    LSolLDiag = -USol[0:-1]/hx
    LSolMDiag = USol[1:]/hx

    N = len(LSolLDiag)
    LUpMDiag = np.zeros((N+2,))
    LUpLDiag = np.zeros((N+1,))
    
    #Now we need to correct for the boundary conditions
    #The entries to the front of MDiag and LDiag account for Robin BC's while
    #the entries to the end of MDiag and UDiag account for Neumann BC's
    LUpMDiag[0] = 0
    LUpMDiag[1:-1] = LSolMDiag 
    LUpMDiag[-1] = 0

    LUpLDiag[0:-1] = LSolLDiag
    LUpLDiag[-1] = 0

    # keyboard
    #Final variable coefficient implicit operator
    #Lup = scipy.sparse.diags([LUpLDiag,LUpMDiag],[-1, 0],shape=(Ncell+2,Ncell+2))
    #Lup = Lup.toarray()

    return LUpLDiag, LUpMDiag


            
#This is the upwinding operator. It is the same for all species. 
UpwindL, UpwindM = UpwindOperatorConstruct()


#Check functionality and speed


#We'll just save these for now (so we can graphically compare at the end)
#f = scipy.interpolate.interp1d(XcellExtend,Hconc)
#startH = f(XedgeExtend)
startH = np.interp(XedgeExtend, XcellExtend, Hconc)

#f = scipy.interpolate.interp1d(XcellExtend,Aconc)
#startA = f(XedgeExtend)
startA = np.interp(XedgeExtend, XcellExtend, Aconc)

#f = scipy.interpolate.interp1d(XcellExtend,Iconc)
#startC = f(XedgeExtend)
startC = np.interp(XedgeExtend, XcellExtend, Iconc)

#f = scipy.interpolate.interp1d(XcellExtend,Bconc)
#startB = f(XedgeExtend)
startB = np.interp(XedgeExtend, XcellExtend, Bconc)
#print(Lhyd)
#input('w')

#print(len(XedgeExtend))
#input('w')

###############################################
###  Change pars here
###############################################
##HydValR = 10*HydValR
##IonValR = 10*IonValR
#
##nominal_pars = np.array([HydValR \
##                        ,IonValR\
##                        ,HydExchangeRate\
##                        ,BicExchangeRate])
#nominal_pars = np.array([-1.227820e+00, -1.560820e+00,\
#        -1.370450e+00,  1.393940e+05,  1.147480e+01, 2.070128e-01])
#
##nominal_pars = np.array([-1.227820e+00, -1.560820e+00,\
##        -1.370450e+00,  1.393940e+05,  1.147480e+01, 2.070128e-01])
#
#nominal_pars = np.array([-1.15411e+00 ,-1.78246e+00 ,-1.15640e+00  ,6.63590e+04  ,5.14826e+00\
# ,-7.26700e-01])
#
#start = time.time()
##Hnow, Anow, Bnow, Cnow, Pnow\
#out, out2 = QOI(nominal_pars\
#    ,rescaled, hx\
#    , Dh, Db, Di, Da\
#    , ValH, ValB, ValI, ValA\
#    , Kbind, Ncell, Nedge\
#    , USol\
#    , BicValR\
#    , Hconc, Iconc, Bconc, Aconc, DPsi\
#    , XcellExtend, XedgeExtend\
#    , ThetaS, ThetaShx\
#    , UpwindL, UpwindM\
#    , data_main, indices, indptr\
#    , startA, startB, startC, startH\
#    , Ncellp2, Nedgep2, Nrows, Ncols\
#    ,Nstart_Hop  ,Nstart_ConH \
#    ,Nstart_DfutH ,Nstart_Aop  \
#    ,Nstart_BonA ,Nstart_DfutA\
#    ,Nstart_AonB ,Nstart_Bop  \
#    ,Nstart_DfutB ,Nstart_HonC \
#    ,Nstart_Cop  ,Nstart_DfutC\
#    , Nstart_eye\
#    , SolVelValL, SolValL\
#    )
#finish = time.time()
#print('\n',finish-start)
#input('w')
#
#input('Press a key for SolnCheck')
#print('Running SolnCheck. Beep bop beep')
#AniValR = HydValR + IonValR - BicValR
#SolnCheck(\
#            Hnow, Anow, Bnow, Cnow, Pnow\
#            , Dh, Da, Db, Di\
#            , ValH, ValA, ValB, ValI\
#            , XcellExtend, XedgeExtend\
#            , ThetaS, hx\
#            , USol\
#            , Kbind, rescaled\
#            , HydExchangeRate\
#            , BicExchangeRate\
#            , HydExchangerParam\
#            , BicExchangerParam\
#            , HydValR, AniValR, BicValR, IonValR\
#            , startH, startA, startB, startC\
#            )
#
#input('w')

#Now lets construct my slightly off-size identity
#EyeDiag = np.ones((Nedge+2,1))

#ModEye = scipy.sparse.spdiags[EyeDiag,0,Nedge+2,Ncell+2]


##--------------------------------------------------------------
## USER DEFINED par nominal values (optional)
##--------------------------------------------------------------
pars  = np.array([1
            ,1
            ,1
            ,1
            ,1
            ]) 


##--------------------------------------------------------------
## USER DEFINED variation around nominal values (optional)
##--------------------------------------------------------------
cov = 0.5

##--------------------------------------------------------------
## bounds for parameters (mandatory)
##--------------------------------------------------------------
#lb = pars - cov*np.abs(pars)
#ub = pars + cov*np.abs(pars)

HEP_lb = HydExchangerParam - cov*np.abs(HydExchangerParam)
HEP_ub = HydExchangerParam + cov*np.abs(HydExchangerParam)

BEP_lb = BicExchangerParam - cov*np.abs(BicExchangerParam)
BEP_ub = BicExchangerParam + cov*np.abs(BicExchangerParam)

# All Pars
#lb = np.array([-4, -4, -6, -6, HEP_lb, BEP_lb, -1])
#ub = np.array([-1, -1, -1, -1, HEP_ub, BEP_ub, 1])

# Source fixed
lb = np.array([-4, -4, -6, -6, HEP_lb, BEP_lb])
ub = np.array([-1, -1, -1, -1, HEP_ub, BEP_ub])

diff_b = ub - lb


N_samples = 50000

N_pars = len(lb)


#p_idx = 2
#grid_idx = 2
#grid_reso = 20
#num_cores = 8

if __name__ == "__main__":
    p_idx = int(sys.argv[1])
    grid_idx = int(sys.argv[2])
    grid_reso = int(sys.argv[3])
    num_cores = int(sys.argv[4])


# 1. Create par list

# Get N samples in P-1 sample space
N_dims = N_pars - 1
file_name = './Sobol_sequences/dimensions_' + str(N_dims) + '.npy'
All_samples = np.load(file_name)
All_samples = All_samples[:N_samples]

all_pars_samples = np.zeros((N_samples, N_pars))

un_p_idx = 0

for p in range(N_pars):
    if p == p_idx:
        continue
    all_pars_samples[:,p] = lb[p] + All_samples[:,un_p_idx]*diff_b[p]
    un_p_idx += 1


all_pars_samples[:, p_idx] = lb[p_idx] + grid_idx/grid_reso*diff_b[p_idx]

print(all_pars_samples)



results = Parallel(n_jobs=num_cores, verbose = 12, backend='multiprocessing')\
                            (delayed(QOI)\
                                (par\
                                 #Auxilliary arguments
                                , rescaled, hx\
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
                                , SolVelValL, SolValL
                                ) for par in all_pars_samples)


results = np.array(results)                                

np.save('pars_'+str(p_idx)+'_'+str(grid_idx)+'.npy', all_pars_samples)
np.save('outs_'+str(p_idx)+'_'+str(grid_idx)+'.npy', results)


