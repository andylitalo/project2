"""
These methods are used to visualize the projected data.
"""

import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


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
    
def plot_movies_three_methods(movie_data, data_all, genre, title_prefix='', x_lim=[-5,5], 
                              y_lim=[-5,5]):
    """
    Plots movies from given genre for all three methods.
    """
    # extract data for each method from list of data
    data_1, data_2, data_3 = data_all
    # get indices of movies in genre
    inds = get_movies_from_genre(movie_data, genre)
    
    # plot movies - method 1
    proj_1 = get_proj(inds, data_1)
    plot_movies(proj_1, title=title_prefix + "method 1")

    # plot movies - method 2
    proj_2 = get_proj(inds, data_2)
    plot_movies(proj_2, title=title_prefix + "method 2")
    
    # plot movies - method 3
    proj_3 = get_proj(inds, data_3)
    plot_movies(proj_3, title=title_prefix + "method 3")
 
    
def plot_genres_three_methods(movie_data, data_all, genre_list, x_lim=[-5,5], y_lim=[-5,5]):
    """
    Plots movies of different genres in different colors for all three methods.
    """
    
    for i in range(len(data_all)):
        data = data_all[i]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # plot movies from each genre in a different color
        for j in range(len(genre_list)):
            genre = genre_list[j]
            # get indices of movies in genre
            inds = get_movies_from_genre(movie_data, genre)
            proj = get_proj(inds, data)
            x = proj[:,0]
            y = proj[:,1]
            ax.plot(x, y, 'o', label=genre)
        title = "Method " + str(i+1) + " - "
        for i in range(len(genre_list)-1):
            title += genre_list[i]
            if i < len(genre_list):
                title += ", "
        ax.set_title(title)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        plt.legend(loc="best")
    
def title_genres(ind, headers):
    """
    Returns string of movie title followed by genres.
    """
    movie_str = movie_data['Movie Title'][ind]
    for i in range(2,len(headers)-1):
        genre = headers[i]
        if movie_data[genre][ind]:
            movie_str += ", " + genre
            
    return movie_str
    
def print_most_extreme(movie_data, data_all, num=5):
    """
    Prints the names of the movies with the most extreme values of each coordinate.
    """
    headers = list(movie_data)
    for i in range(len(data_all)):
        print("")
        print("Method %i" %(i+1))
        # select data for given method
        data = data_all[i]

        # greatest positive x value
        print("greatest positive x")
        sorting_inds_x = np.argsort(data[0,:])
        inds_x_pos = sorting_inds_x[-num:]
        for j in inds_x_pos:
            print(title_genres(j, headers))
        
        # smallest negative x value
        print("lowest negative x")
        inds_x_neg = sorting_inds_x[:num]
        for j in inds_x_neg:
            print(title_genres(j, headers))
            
        # greatest positive y value
        print("greatest positive y")
        sorting_inds_y = np.argsort(data[1,:])
        inds_y_pos = sorting_inds_y[-num:]
        for j in inds_y_pos:
            print(title_genres(j, headers))
            
            
        # lowest negative y value
        print("lowest negative y")
        inds_y_neg = sorting_inds_y[:num]
        for j in inds_y_neg:
            print(title_genres(j, headers))
            
        # closest to 0
        print("closest to 0")
        sorting_inds_radii = np.argsort(np.sqrt((data[0,:])**2+(data[1,:])**2))
        inds_near_zero = sorting_inds_radii[:num]
        for j in inds_near_zero:
            print(title_genres(j, headers))
            
        # farthest from 0
        print("farthest from 0")
        inds_far_zero = sorting_inds_radii[-num:]
        for j in inds_far_zero:
            print(title_genres(j, headers))
        
        
def order_genres_by_variance(movie_data, data_all):
    """
    Returns list of genres in order of decreasing variance for each method.
    """
    # get list of genres
    genre_list = list(movie_data)[2:-1]
    # initialize result, list of lists, each list is for one method
    result = []
    
    # loop through each method
    for data in data_all:
        # array of variances
        var_arr = np.zeros([len(genre_list)])
        for i in range(len(genre_list)):
            genre = genre_list
            inds = get_movies_from_genre(movie_data, genre)
            proj = get_proj(inds, data)
            (xc, yc) = np.mean(proj, axis=0)
#            
#            x = proj[:,0]
#            y = proj[:,1]
#            r = np.sqrt((x-xc)**2 + (y-yc)**2)
#            var_arr[i] = np.mean(r)
            
            # how best to estimate variance?
            pca = PCA(n_components=2)
            pca.fit(proj)
            var_arr[i] = pca.singular_values_[0]
        
        # sort by variance
        inds = np.argsort(var_arr)[::-1]
        sorted_genre_list = [genre_list[ind] for ind in inds]
        result+= [sorted_genre_list]      
           
    return result
# main function for running methods
if __name__=='__main__':
    # load projected data (2 x M and 2 x N)
    users_1, users_2, users_3, movies_1, movies_2, movies_3 = load_projections()
    
    # load movies and ratings
    movie_data, rating_data = load_data()
    
    data_all = [movies_1, movies_2, movies_3]
    genre_lists = order_genres_by_variance(movie_data, data_all)
    
    for i in range(len(genre_lists[1])):
        genre = [genre_lists[1][i]]
        plot_genres_three_methods(movie_data, [data_all[1]], genre)
#    # plot top 10 movies (ratings)
#    inds_best_ten = best_ten_movies(movie_data, rating_data)
#    best_ten_proj_1 = get_proj(inds_best_ten, movies_1)
#    plot_movies(best_ten_proj_1, title='top 10 - method 1')
#    
#    # plot top 10 movies (ratings) - method 2
#    inds_best_ten = best_ten_movies(movie_data, rating_data)
#    best_ten_proj_2 = get_proj(inds_best_ten, movies_2)
#    plot_movies(best_ten_proj_2, title='top 10 - method 2')
#    
#    # plot top 10 movies (ratings) - method 3
#    best_ten_proj_3 = get_proj(inds_best_ten, movies_3)
#    plot_movies(best_ten_proj_3, title='top 10 - method 3')
#    
#    # plot most popular 10 movies (popularity)
#    inds_pop_ten = pop_ten_movies(rating_data)    
#
#    pop_ten_proj_1 = get_proj(inds_pop_ten, movies_1)
#    plot_movies(pop_ten_proj_1)
#
#    movies_all = [movies_1, movies_2, movies_3]
#    # compare genres
#    plot_genres_three_methods(movie_data, movies_all, ["Musical", "Horror",
#                                                       "Childrens", "Western",
#                                                       "Animation"])
#    
#    # compare genres
#    plot_genres_three_methods(movie_data, movies_all, ["Drama", "Comedy",
#                                                       "Childrens"])
#    
#    # print names of movies with most extreme values
#    print_most_extreme(movie_data, movies_all)
