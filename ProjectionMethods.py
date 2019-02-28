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
        params = pkl.load(f)
        
    U1, V1, RMSE1 = params['method 1']
    U2, V2, a2, b2, RMSE2 = params['method 2']
    U3, V3, a3, b3, RMSE3 = params['method 3']
 
    return U1, V1, U2, V2, a2, b2, U3, V3, a3, b3


def mean_center(V):
    """
    Centers the mean of each row of V about 0
    """
    return V - np.transpose(np.array([np.mean(V, axis=1)]))

def step_2a(V, mean_center=False):
    """
    Step 2a does SVD of V and returns the first two columns of the left matrix.
    """
    if mean_center:
        # mean center V
        V = mean_center(V)
    # SVD of V matrix from matrix factorization and project for each method
    A, Sigma, B = np.linalg.svd(V)
    A_2d = A[:,0:2]
    
    return A_2d


def step_2b(U, V, A_2d):
    """
    Project every movie and user onto the first two columns of A.
    """
    U_proj = np.matmul(np.transpose(A_2d),U)
    V_proj = np.matmul(np.transpose(A_2d),V)
    
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
    # load parameters: U is k x M, V is k x N, a is M x 1, b is N x 1
    U1, V1, U2, V2, a2, b2, U3, V3, a3, b3 = load_matrix_factorization()
    
    # step 2a: return first two columns of left matrix in SVD of V, N x 2
    A1 = step_2a(V1)
    A2 = step_2a(V2)
    A3 = step_2a(V3)
    
    # step 2b: project all movies and users onto first two columns of A
    # U_proj is 2 x M and V_proj is 2 x N
    U1_proj, V1_proj = step_2b(U1, V1, A1)
    U2_proj, V2_proj = step_2b(U2, V2, A2)
    U3_proj, V3_proj = step_2b(U3, V3, A3)
    
    # step 2c: normalize to unit variance
    U1_norm, V1_norm = step_2c(U1_proj, V1_proj)
    U2_norm, V2_norm = step_2c(U2_proj, V2_proj)
    U3_norm, V3_norm = step_2c(U3_proj, V3_proj)
    
    # save data
    params = {}
    params['projected users'] = [U1_norm, U2_norm, U3_norm]
    params['projected movies'] = [V1_norm, V2_norm, V3_norm]
    
    with open('data/projection_data.pkl', 'wb') as f:
        pkl.dump(params, f)
