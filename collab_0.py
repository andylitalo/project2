import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    
    DU = reg*Ui - 2*Vj*(Yij - np.dot(Ui,Vj))  
    return eta*DU

    pass   

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    
    DV = reg*Vj - 2*Ui*(Yij - np.dot(Ui,Vj)) 
    return eta*DV

    pass

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    
    err = 0
    for m in np.arange(0,Y.shape[0]):
        i = Y[m,0]-1
        j = Y[m,1]-1
        Y_ij = Y[m,2]
        
        err += (Y_ij - np.dot(U[i,:],V.T[:,j]))**2
    
    #err += reg*(np.linalg.norm(U)**2 + np.linalg.norm(V)**2)
    
    return 0.5*err/float(Y.shape[0])

    pass


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, checkpoints=None):
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
    Checkpoints is the list of epochs that we want to measure the error at.
    If this optional parameter is used, we only return (Us, Vs, epochs), which
    are lists of U and V matrices at the designated epochs.
    """
    
    # Generate the initial weights
    U = np.random.uniform(-0.5,0.5,(M,K))
    V = np.random.uniform(-0.5,0.5,(N,K))
    
    # Get the size of the dataset
    N_data = Y.shape[0];
    
    # Keep track of errors at each epoch
    err_trace = np.zeros(max_epochs+1)
    err_trace[0] = get_err(U,V,Y,reg)

    # Remember Us, Vs at checkpoints
    Us = []
    Vs = []
    epochs = []

    for s in np.arange(0,max_epochs):
        # Shuffle the data
        perm = np.random.permutation(N_data);
        Y = Y[perm,:]
        
        for m in np.arange(0,N_data):
            # Get the indices/values
            i = Y[m,0]-1
            j = Y[m,1]-1
            Y_ij = Y[m,2]
 
            # Perform the SGD update
            U[i,:] -= grad_U(U[i,:], Y_ij, V.T[:,j], reg, eta)
            V.T[:,j] -= grad_V(V.T[:,j], Y_ij, U[i,:], reg, eta)
    
        # Record the error
        err = get_err(U,V,Y,reg)        
        err_trace[s+1] = err;

        print("Epoch number", s)
        print("Reg error", err)
        
        # Check if stopping criterion satisfied
        if np.abs((err_trace[s]-err_trace[s+1])/(err_trace[0]-err_trace[1])) < eps:
            Us.append(np.copy(U))
            Vs.append(np.copy(V))
            epochs.append(s)
            break

        if checkpoints is not None and s in checkpoints:
            Us.append(np.copy(U))
            Vs.append(np.copy(V))
            epochs.append(s)

    if checkpoints is None:
        return U,V,get_err(U, V, Y)
    else:
        return Us, Vs, epochs


