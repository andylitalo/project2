"""
These methods are used to visualize the projected data.
"""

import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_projections(filepath='data/projection_data.pkl'):
    """
    Loads projected data of users and movies.
    """
    with open(filepath, 'rb') as f:
        params = pkl.load(f)
        
    U1_proj, U2_proj, U3_proj = params['projected users']
    V1_proj, V2_proj, V3_proj = params['projected movies']
    
    return U1_proj, U2_proj, U3_proj, V1_proj, V2_proj, V3_proj

def load_data(movie_filepath='data/movies.txt', rating_filepath='data/data.txt'):
    """
    Returns dataframes of movie genre data and of ratings.
    """
    # Movie metadata
    movie_data = pd.read_csv(movie_filepath, sep='\t', names=['Movie Id', 'Movie Title', 'Unknown',
                                                              'Action', 'Adventure', 'Animation',
                                                              'Children''s', 'Comedy', 'Crime', 
                                                              'Documentary','Drama', 'Fantasy', 
                                                              'Film-Noir', 'Horror', 'Musical',
                                                              'Mystery', 'Romance', 'Sci-Fi', 
                                                              'Thriller', 'War', 'Western'])
    # Ratings
    rating_data = pd.read_csv(rating_filepath, sep='\t', 
                              names=['User Id', 'Movie Id', 'Rating'])
    
    # number of movies
    n_movies = len(movie_data)
    # initialize array to store mean ratings
    mean_ratings = np.zeros([n_movies])
    
    # compute mean rating
    for i in range(n_movies):
        movie_id = i+1
        matching_id = np.where(rating_data['Movie Id']==movie_id)[0]
        ratings = rating_data['Rating'][matching_id]
        mean_ratings[i] = np.mean(ratings)
        
    # save mean ratings
    movie_data['Mean Rating'] = mean_ratings

    return movie_data, rating_data

def best_ten_movies(movie_data, rating_data):
    """
    Returns the indices of the ten highest-rated movies.
    """
    # get array of movie ids and frequency of ratings
    movies, ret = np.unique(rating_data['Movie Id'], return_counts=True)
    # get top ten rated movies' ratings
    # get indices that sort by number of ratings, low to high
    sorting_inds = np.argsort(movie_data['Mean Rating'])
    # indices of top 10 are at the end of the sorting list
    inds_best_ten = sorting_inds[-10:][::-1]
    
    return inds_best_ten.values

def pop_ten_movies(rating_data):
    """
    Returns the indices of the ten most popular movies.
    """
    # get array of movie ids and frequency of ratings
    movies, rating_freq = np.unique(rating_data['Movie Id'], return_counts=True)
    # get indices that sort by number of ratings, low to high
    sorting_inds = np.argsort(rating_freq)
    # indices of top 10 are at the end of the sorting list
    inds_pop_ten = sorting_inds[-10:][::-1]
    
    return inds_pop_ten

def get_movies_from_genre(movie_data, genre):
    """
    Returns indices of movies of a certain genre.
    """
    return np.where(movie_data[genre].values==1)[0]

def get_proj(inds, proj):
    """
    Returns the projection of the given indices.
    """
    return np.array([proj[:,i] for i in range(len(proj[0,:])) if i in inds])
    
def plot_movies(data, title='', x_lim=[-5,5], y_lim=[-5,5]):
    """
    Plots movie projections in standard format for consistency.
    """
    x = data[:,0]
    y = data[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'o')
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
# main function for running methods
if __name__=='__main__':
    # load projected data (2 x M and 2 x N)
    users_1, users_2, users_3, movies_1, movies_2, movies_3 = load_projections()
    
    # load movies and ratings
    movie_data, rating_data = load_data()
    
    # plot top 10 movies (ratings)
    inds_best_ten = best_ten_movies(movie_data, rating_data)
    best_ten_proj_1 = get_proj(inds_best_ten, movies_1)
    plot_movies(best_ten_proj_1)
    
    # plot most popular 10 movies (popularity)
    inds_pop_ten = pop_ten_movies(rating_data)
    pop_ten_proj_1 = get_proj(inds_pop_ten, movies_1)
    plot_movies(pop_ten_proj_1)
    
    # plot horror movies
    inds_horror = get_movies_from_genre(movie_data, 'Horror')
    horror_proj_1 = get_proj(inds_horror, movies_1)
    plot_movies(horror_proj_1)
    
    # plot musical movies
    inds_musical = get_movies_from_genre(movie_data, 'Musical')
    musical_proj_1 = get_proj(inds_musical, movies_1)
    plot_movies(musical_proj_1)
    
    # plot musical movies
    inds_musical = get_movies_from_genre(movie_data, 'Musical')
    musical_proj_3 = get_proj(inds_musical, movies_3)
    plot_movies(musical_proj_3)
#    plt.plot(users_1[0,:], users_1[1,:], 'o')
#    
#    plt.figure()
#    plt.plot(movies_2[0,:], movies_2[1,:], 'o')
