# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:21:31 2019

@author: Josef M Sabuda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

def get_RMSE_off(model, testset):
    """
    Returns the RMSE of an off-the-shelf model over the test set.
    """
    # predict ratings for the testset
    predictions = model.test(testset)

    # Then compute RMSE
    RMSE = accuracy.rmse(predictions)
    
    return RMSE

def method_23_off(trainset, testset, k=20, eta=0.03, n_epochs=100, 
                  reg=0.1, reg_bi=0, reg_bu=0, mu=0):
    """
    Uses off-the-shelf method with bias but no regularization over the bias.
    """
    # set up model
    model = SVD(n_factors=k, n_epochs=n_epochs, biased=True, lr_all=eta, 
                     reg_all=reg, reg_bi=reg_bi, reg_bu=reg_bu, verbose=False)
    
    # train model
    model.fit(trainset)
    
    # get matrices
    U = model.pu
    V = model.qi
    ai = model.bu
    bj = model.bi
    
    # compute RMSE
    RMSE = get_RMSE_off(model, testset)
    
    return U, V, ai, bj, RMSE

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
    
def plot_movies(data, movie_data, title='', indices='',x_lim=[-5,5], y_lim=[-5,5]):
    """
    Plots labeled movie projections in standard format for consistency.
    """
    titles = movie_data['Movie Title'][indices]
    titles = list(titles)
    x = data[:,0]
    y = data[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'o')
    ax.set_title(title)
    #ax.set_xlim(x_lim)
    #ax.set_ylim(y_lim)
    for txt in range(len(x)):
        ax.annotate(titles[txt],((x[txt],y[txt])))
    
# load training data
Y_train = np.loadtxt('data/train.txt').astype(int)
Y_test = np.loadtxt('data/test.txt').astype(int)
# also load using surprise library for compatibility with surprise packages
data = Dataset.load_builtin('ml-100k')
# sample random trainset and testset
# test set is made of 10% of the ratings.
trainset, testset = train_test_split(data, test_size=.10)

#Method 3: add bias and reg
U3, V3_trans, a3, b3, RMSE3 = method_23_off(trainset, testset, reg_bu=0.1, reg_bi=0.1)
U3_trans = np.transpose(U3)
V3 = np.transpose(V3_trans)

#Modify output data

#Mean Center the data
temp_means = np.zeros(V3.shape[0])
for i in range(V3.shape[0]):
    temp_means[i] = np.mean(V3[i,:])
for i in range(V3.shape[0]):
    for j in range(V3.shape[1]):
        V3[i,j] -= temp_means[i]

#Do SVD of V and return the first two columns of the left matrix.
A, Sigma, B = np.linalg.svd(V3)
A_2d = A[:,0:2]
A3 = A_2d

#Project U3,V3 onto A3
U3_proj = np.matmul(np.transpose(A3),U3_trans)
V3_proj = np.matmul(np.transpose(A3),V3)

#Rescale U3 and V3 to have unit variance in each of the 2 plotted dimensions.
U3_norm = np.copy(U3_proj)
V3_norm = np.copy(V3_proj)

temp_U3_stds = np.zeros(U3_proj.shape[0])
temp_V3_stds = np.zeros(V3_proj.shape[0])

for i in range(V3_proj.shape[0]):
    temp_V3_stds[i] = np.std(V3_proj[i,:])
for i in range(V3_proj.shape[0]):
    for j in range(V3_proj.shape[1]):
        V3_norm[i,j] = V3_norm[i,j]/temp_V3_stds[i]

for i in range(U3_proj.shape[0]):
    temp_U3_stds[i] = np.std(U3_proj[i,:])
for i in range(U3_proj.shape[0]):
    for j in range(U3_proj.shape[1]):
        U3_norm[i,j] = U3_norm[i,j]/temp_U3_stds[i]

#Start visualizing
users_3 = U3_norm
movies_3 = V3_norm
    
# load movies and ratings
movie_data, rating_data = load_data()

#List of genres
#'Action', 'Adventure', 'Animation','Children''s', 'Comedy', 'Crime', 'Documentary','Drama', 'Fantasy', 
#'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'

'''
# plot top 10 movies (ratings) - method 3
inds_best_ten = best_ten_movies(movie_data, rating_data)
best_ten_proj_3 = get_proj(inds_best_ten, movies_3)
plot_movies(best_ten_proj_3, movie_data, indices = inds_best_ten, title='top 10 - method 3')
    
# plot most popular 10 movies (popularity)
inds_pop_ten = pop_ten_movies(rating_data)
pop_ten_proj_1 = get_proj(inds_pop_ten, movies_3)
plot_movies(pop_ten_proj_1, movie_data, indices = inds_pop_ten, title='top 10 - most popular')
'''

N = 20
Fant_movie_indices = get_movies_from_genre(movie_data, "Fantasy")
fantasy_proj_1 = get_proj(Fant_movie_indices[0:N], movies_3)
plot_movies(fantasy_proj_1, movie_data, indices = Fant_movie_indices[0:N], title='Some Fantasy Movies')



#Nearest neighbors
def nearest_n_neighbors(N, movie_index, mult_movie_indices, proj):
    """
    Returns the indices of the nearest n neighbors to a given movie from the 
    movies indexed in mult_movie_indices.
    """
    
    if N > len(mult_movie_indices):
        N = len(mult_movie_indices)
        
    x = proj[0,movie_index]
    y = proj[1,movie_index]
    
    dist = -1*np.ones(len(mult_movie_indices))
    for i in range(len(mult_movie_indices)):
        x2 = proj[0,mult_movie_indices[i]]
        y2 = proj[1,mult_movie_indices[i]] 
        dist[i] = np.sqrt((x-x2)**2 + (y-y2)**2)
    
    nearest_neighbor_indices = np.zeros(N)
    for i in range(N):
        min_dist = 1000
        index = 1000
        for j in range(len(dist)):
            if dist[j] < min_dist:
                min_dist = dist[j]
                index = j
        nearest_neighbor_indices[i] = index
        dist[index] = 1000
        
    return(nearest_neighbor_indices)   

#Example Nearest 5 neighbors to a given movie
inds_pop_ten = pop_ten_movies(rating_data)
title_most_pop = movie_data['Movie Title'][inds_pop_ten[0]]
ind_most_pop = inds_pop_ten[0]

horror_indices = get_movies_from_genre(movie_data, 'Horror')
nearest_horror_to_star_wars = nearest_n_neighbors(5, ind_most_pop, horror_indices, movies_3)

print(nearest_horror_to_star_wars)

indices = np.ones(len(nearest_horror_to_star_wars)+1)
for i in range(len(nearest_horror_to_star_wars)):
    indices[i] = nearest_horror_to_star_wars[i]
indices[-1] = ind_most_pop
fantasy_proj_1 = get_proj(indices, movies_3)
plot_movies(fantasy_proj_1, movie_data, indices = indices, title='Star wars and closest horror movies')
