#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_rand_score as ARI
import dcsbm

## PARSER to give parameter values
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

## Set tolerance
parser.add_argument("-t","--tol", type=float, dest="tolerance", default=1e-5, const=True, nargs="?",\
	help="Float: convergence criterion (relative difference in average predictive log-likelihood)")
## Set maximum number of iterations
parser.add_argument("-i","--maxiter", type=int, dest="max_iter", default=150, const=True, nargs="?",\
	help="Integer: maximum number of iterations for the variational inference algorithm")
## Model type
parser.add_argument("-m", "--model", dest="model", default = 2, type=int,
	help="Model: 0 for standard, 1 for normalised, 2 for spherical.")
## Initial dimension of the embedding
parser.add_argument("-d", "--dimension", dest="dimension", default=50, type=int,
	help="Initial dimension of the embedding.")
## Graph type
parser.add_argument("-g", "--graph", dest="graph", default = 'icl2', type=str,
	help="Type of graph: icl1, icl2 or icl3.")
## Use the approximated inference procedure
parser.add_argument("-a", "--approx", dest="approx",
	action='store_true', help='Use the approximated inference procedure (fast).')

## Parse arguments
args = parser.parse_args()
tolerance = args.tolerance
max_iter = args.max_iter
model = args.model
graph = args.graph
m = args.dimension

## Import data
if graph == 'icl1':
    true_labs = np.loadtxt('Data/labs1.csv', delimiter=',')
    X = np.load('Data/X_ICL1.npy')
elif graph == 'icl2':
    true_labs = np.loadtxt('Data/labs2.csv', delimiter=',')
    X = np.load('Data/X_ICL2.npy')
elif graph == 'icl3':
    true_labs = np.loadtxt('Data/labs3.csv', delimiter=',')
    X = np.load('Data/X_ICL3.npy')
else:
    raise ValueError('Invalid graph.')

## Obtain X_tilde
X_tilde = np.divide(X, np.linalg.norm(X,axis=1)[:,np.newaxis])
## Obtain Theta
Theta = dcsbm.theta_transform(X)

## Calculate the BIC for some combinations of models
mod = ['X','X_tilde','Theta','SCORE'][model]
## Determine the model type
if mod == 'Theta':
    mod_type = 'theta'
elif mod == 'X_tilde':
    mod_type = 'normalised'
elif mod == 'SCORE':
    mod_type = 'score'
else:
    mod_type = None

## BIC and ARI
bic = np.zeros((30,19))
ari = np.zeros((30,19))
for d in range(1,31):
    for K in range(2,21):
        ## Initialise the model
        M = dcsbm.EGMM(K=K)
        if args.approx:
            try:
                print('\rd: '+str(d)+'\tK: '+str(K), end='')
                z = M.fit_predict_approximate(X,d=d,transformation=mod_type)
            except:
                print(str(d), str(K))
                raise ValueError('Error')
        else:
            print('\rd: '+str(d)+'\tK: '+str(K), end='')
            z = M.fit_predict(X,d=d,transformation=mod_type,verbose=False,random_init=False,max_iter=max_iter,tolerance=tolerance)
        ## Update the value of the BIC
        bic[d-1,K-2] = M.BIC()
        ari[d-1,K-2] = ARI(true_labs,z)

## Save files
np.savetxt('bic_'+graph+'_'+mod+'_'+str(m)+'.csv', bic, delimiter=',', fmt='%.4f')
np.savetxt('ari_'+graph+'_'+mod+'_'+str(m)+'.csv', ari, delimiter=',', fmt='%.5f')