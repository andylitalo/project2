"""
These methods are used to visualize the projected data.
"""

import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt


def load_projections(filepath='data/projection_data.pkl'):
    """
    Loads projected data of users and movies.
    """
    with open(filepath, 'rb') as f:
        params = pkl.load(f)
        
    U_proj = params['projected users']
    V_proj = params['projected movies']
    
    return U_proj, V_proj


# main function for running methods
if __name__=='__main__':
    # load projected data (2 x M and 2 x N)
    users, movies = load_projections()
    
    plt.plot(users[0,:], users[1,:], 'o')
    
    plt.figure()
    plt.plot(movies[0,:], movies[1,:], 'o')
