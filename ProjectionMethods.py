"""
Methods for visualizing data (step 2 of project 2).
"""

import seaborn as sns
import pickle as pkl
import numpy as np


def load_matrix_factorization(filepath='data/matrix_factorization.pkl'):
    """
    Loads the parameters of the matrix factorization.
    """
    with open(filepath, 'rb') as f:
        params_dict = pkl.load(f)
        
    U = params_dict['U']
    V = params_dict['V']
    a = params_dict['a']
    b = params_dict['b']
    
    return U, V, a, b


def mean_center(V):
    """
    Centers the mean of each row of V about 0
    """
    return V - np.transpose(np.array([np.mean(V, axis=1)]))


def step_2a(V):
    """
    Step 2a does SVD of V and returns the first two columns of the left matrix.
    """
    # SVD of V matrix from matrix factorization
    A, Sigma, B = np.linalg.svd(V)
    
    # get the number of dimensions of the matrix factorization
    k = len(B)
    
    # first two columns of A^T
    A12 = A[:k,0:2]
    
    return A12


def step_2b(U, V, A12):
    """
    Project every movie and user onto the first two columns of A.
    """
    U_proj = np.matmul(np.transpose(A12),np.transpose(U))
    V_proj = np.matmul(np.transpose(A12),np.transpose(V))
    
    return U_proj, V_proj

def step_2c(U_proj, V_proj):
    """
    Rescales U and V to have unit variance in each of the 2 plotted dimensions.
    """
    # compute standard deviations
    U_std = np.std(U_proj, axis=1)
    V_std = np.std(V_proj, axis=1)
    
    # normalize
    U_norm = U_proj/np.transpose(np.array([U_std]))
    V_norm = V_proj/np.transpose(np.array([V_std]))
    
    return U_norm, V_norm


# main function to run step 2
if __name__=='__main__':
    # load parameters: U is M x k, V is N x k, a is M x 1, b is N x 1
    U, V, a, b = load_matrix_factorization()
    
    # step 2a: return first two columns of left matrix in SVD of V, N x 2
    A12 = step_2a(V)
    # step 2b: project all movies and users onto first two columns of A
    # U_proj is 2 x M and V_proj is 2 x N
    U_proj, V_proj = step_2b(U, V, A12)
    # step 2c: normalize to unit variance
    U_norm, V_norm = step_2c(U_proj, V_proj)
    
    # save data
    params = {}
    params['projected users'] = U_norm
    params['projected movies'] = V_norm
    
    with open('data/projection_data.pkl', 'wb') as f:
        pkl.dump(params, f)
