"""
Use this to transpose U and V matrices saved by previous versions of MatrixFactorizionMethods.py for use in ProjectionMethods.py
"""

def transpose_matrices(filepath='data/matrix_factorization.pkl'):
    """
    Loads the parameters of the matrix factorization.
    """
    with open(filepath, 'rb') as f:
        params = pkl.load(f)
        
    U1, V1, RMSE1 = params['method 1']
    U2, V2, a2, b2, RMSE2 = params['method 2']
    U3, V3, a3, b3, RMSE3 = params['method 3']
 
    # save results
    params = {}
    params['method 1'] = [np.transpose(U1), np.transpose(V1), RMSE1]
    params['method 2'] = [np.transpose(U2), np.transpose(V2), a2, b2, RMSE2]
    params['method 3'] = [np.transpose(U3), np.transpose(V3), a3, b3, RMSE3]
    
    with open('data/matrix_factorization.pkl', 'wb') as f:
        pkl.dump(params, f)
        
if __name__=='__main__':
  transpose_matrices()
