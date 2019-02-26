# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta, ai=0, bj=0):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    grad = reg*Ui - 2*Vj*(Yij-(np.dot(Ui,Vj) + ai + bj))
    
    return eta*grad

def grad_V(Vj, Yij, Ui, reg, eta, ai=0, bj=0):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    grad = reg*Vj - 2*Ui*(Yij - (np.dot(Ui,Vj) + ai + bj))
    
    return eta*grad

def grad_a(Ui, Vj, Yij, eta, ai, bj):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate), as well as bias terms ai for users and bj for
    movies.

    Returns the gradient of the regularized loss function with
    respect to ai multiplied by eta.
    """
    grad =  - 2*(Yij - (np.dot(Ui,Vj) + ai + bj))
    
    return eta*grad

def grad_b(Ui, Vj, Yij, eta, ai, bj):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate), as well as bias terms ai for users and bj for
    movies.

    Returns the gradient of the regularized loss function with
    respect to bj multiplied by eta.
    """
    grad =  - 2*(Yij - (np.dot(Ui,Vj) + ai + bj))
    
    return eta*grad

def get_RMSE(U, V, Y, bias=False, a=None, b=None):
    """
    Returns the root-mean-square error of the matrix factorization U, V over Y.
    """
    # number of samples
    num_samples = len(Y)
    
    # calculate squared error term by sum loss for each entry of Y
    sq_err = 0
    for ind in range(len(Y)):
        # extract parameters (i and j are 1-indexed for some reason...)
        i = Y[ind,0]-1
        j = Y[ind,1]-1
        Yij = Y[ind,2]
        Ui = U[i,:]
        Vj = V[j,:]
        if bias:
            ai = a[i]
            bj = b[j]
        else:
            ai = 0
            bj = 0
        # add term
        sq_err += (Yij - (np.dot(Ui,Vj) + ai + bj))**2
    
    RMSE = np.sqrt(sq_err/num_samples)
    
    return RMSE

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, get_all_RMSE=False,
                bias=False):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    # initialize U and V on uniform interval from [-0.5,0.5]
    U = np.random.rand(M,K) - 0.5
    V = np.random.rand(N,K) - 0.5
    if bias:
        a = np.random.rand(M) - 0.5
        b = np.random.rand(N) - 0.5
    else:
        a = None
        b = None
    
    # compute initial error
    RMSE = get_RMSE(U, V, Y, bias=bias, a=a, b=b)
    
    if get_all_RMSE:
        RMSE_list = [RMSE]
    
    # initialize loss reduction
    loss_red_01 = 0
    loss_red = 0
    
    # list of indices to shuffle
    inds = np.arange(len(Y))
    
    # minimize loss
    for epoch in range(max_epochs):
        # shuffle indices
        np.random.shuffle(inds)
        # perform SGD
        for ind in inds:
            i = Y[ind,0]-1
            j = Y[ind,1]-1
            Yij = Y[ind,2]
            Ui = U[i,:]
            Vj = V[j,:]
            if bias:
                ai = a[i]
                bj = b[j]
            else:
                ai = 0
                bj = 0
            # update points
            U[i,:] -= grad_U(Ui, Yij, Vj, reg, eta, ai=ai, bj=bj)
            V[j,:] -= grad_V(Vj, Yij, Ui, reg, eta, ai=ai, bj=bj)
            if bias:
                a[i] -= grad_a(Ui, Vj, Yij, reg, ai, bj)
                b[j] -= grad_b(Ui, Vj, Yij, reg, ai, bj)
            
        # compute new error
        RMSE_new = get_RMSE(U, V, Y, bias=bias, a=a, b=b)
        # store loss reduction from first epoch
        if epoch==0:
            loss_red_01 = RMSE - RMSE_new
        # store loss reduction
        loss_red = RMSE-RMSE_new
        # update error
        if get_all_RMSE:
            RMSE_list += [RMSE_new]
            
        RMSE = RMSE_new
        # check if error reduced by more than tolerance
        if (loss_red / loss_red_01 <= eps):
            break
    
    if get_all_RMSE:
        RMSE = RMSE_list
       
    if bias:
        return U, V, a, b, RMSE
    else:
        return U, V, RMSE
