"""
These methods are used to visualize the projected data.
"""

import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import cm

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
 
  

def plot_genres(movie_data, data, genre_list, method_num, x_lim=[-3,3], y_lim=[-3,3],
                plot_cov=False, plot_ellipse=False, save_plots=False, save_folder='figures/', 
                ext='.png'):
    """
    Plots movies of all genres on different plots for different methods.
    """
       
    # plot movies from each genre in a different color
    for j in range(len(genre_list)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        genre = genre_list[j]
        # get indices of movies in genre
        inds = get_movies_from_genre(movie_data, genre)
        proj = get_proj(inds, data)
        x = proj[:,0]
        y = proj[:,1]
        scatter, = ax.plot(x, y, 'o', label=genre, zorder=1)
        if plot_cov:
            xc, yc, lambda_, eig_vecs, var = get_cov_eig(movie_data,data, genre)
            scatter.set_label(genre + ", %.3f" %var)
            # first eigenvector
            ax.arrow(xc, yc, lambda_[0]*eig_vecs[:,0][0], 
                     lambda_[0]*eig_vecs[:,0][1], width=0.05, color='k',
                     zorder=2)
            # second eigenvector
            ax.arrow(xc, yc, lambda_[1]*eig_vecs[:,1][0], 
                     lambda_[1]*eig_vecs[:,1][1], width=0.05, color='k',
                     zorder=3)
            if plot_ellipse:
                plot_cov_ellipse(xc, yc, lambda_, eig_vecs, ax=ax)
            
             # format plot   
            title = "Method " + str(method_num) + " - " + genre
            ax.set_title(title, fontsize=20)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            plt.legend(loc="best", fontsize=16)
            if save_plots:
                save_name = "method_%i_genre_%s" % (method_num, genre)
                if plot_cov:
                    save_name += "_cov_%i" % (int(1000*var))
                plt.savefig(save_folder + save_name + ext, bbox_inches="tight")
    
    
def get_cov_eig(movie_data, data, genre):
    """
    Returns eigen values and eigenvectors of covariance matrix
    """
    cov_mat = get_cov_mat(movie_data, data, genre)
    var = np.linalg.det(cov_mat)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    x = data[:,0]
    y = data[:,1]
    xc = np.mean(x)
    yc = np.mean(y)
    lambda_ = np.sqrt(eig_vals)
    
    return xc, yc, lambda_, eig_vecs, var

def get_title_genres(ind, headers):
    """
    Returns string of movie title followed by genres.
    """
    movie_str = movie_data['Movie Title'][ind]
    for i in range(2,len(headers)-1):
        genre = headers[i]
        if movie_data[genre][ind]:
            movie_str += ", " + genre
            
    return movie_str
    
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
            print(get_title_genres(j, headers))
        
        # smallest negative x value
        print("lowest negative x")
        inds_x_neg = sorting_inds_x[:num]
        for j in inds_x_neg:
            print(get_title_genres(j, headers))
            
        # greatest positive y value
        print("greatest positive y")
        sorting_inds_y = np.argsort(data[1,:])
        inds_y_pos = sorting_inds_y[-num:]
        for j in inds_y_pos:
            print(get_title_genres(j, headers))
            
            
        # lowest negative y value
        print("lowest negative y")
        inds_y_neg = sorting_inds_y[:num]
        for j in inds_y_neg:
            print(get_title_genres(j, headers))
            
        # closest to 0
        print("closest to 0")
        sorting_inds_radii = np.argsort(np.sqrt((data[0,:])**2+(data[1,:])**2))
        inds_near_zero = sorting_inds_radii[:num]
        for j in inds_near_zero:
            print(get_title_genres(j, headers))
            
        # farthest from 0
        print("farthest from 0")
        inds_far_zero = sorting_inds_radii[-num:]
        for j in inds_far_zero:
            print(get_title_genres(j, headers))
       
        
def get_cov_mat(movie_data, data, genre):
    """
    Returns covariance matrix of movie data from specific genre.
    """
    inds = get_movies_from_genre(movie_data, genre)
    proj = get_proj(inds, data)
    
    return np.cov(np.transpose(proj))

        
def order_genres_by_variance(movie_data, data_all):
    """
    Returns list of genres in order of decreasing variance for each method.
    """
    # get list of genres
    genre_list = list(movie_data)[2:-1]
    # initialize result, list of lists, each list is for one method
    sorted_genre_lists = []
    genre_vars = []
    
    # loop through each method
    for data in data_all:
        # array of variances
        var_arr = np.zeros([len(genre_list)])
        for i in range(len(genre_list)):
            genre = genre_list[i]
            # estimate variance with determinant of covariance matrix
            cov_mat = get_cov_mat(movie_data, data, genre)
            var_arr[i] = np.linalg.det(cov_mat)
        
        # sort by variance
        inds = np.argsort(var_arr)[::-1]
        sorted_genre_list = [genre_list[ind] for ind in inds]
        sorted_genre_lists += [sorted_genre_list] 
        genre_vars += [var_arr]
           
    return sorted_genre_lists, genre_vars


def plot_cov_ellipse(xc, yc, lambda_, eig_vecs, ax=None, label="", color='k'):
    """
    Adds plot of ellipse marking 1 standard deviation to ax.
    Requires that "data" is passed as a 2 x N array.
    """
    ell = Ellipse(xy=(xc, yc), width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=-np.rad2deg(np.arctan2(eig_vecs[0, 1],eig_vecs[0, 0])), 
                  edgecolor=color, fc='None', lw=2, label=label)
    ell.set_facecolor('none')
    # plot
    if not ax:
        ax = plt.subplot(111, aspect='equal')
    ax.add_artist(ell)
    ax.add_patch(ell)


def plot_genre_ellipses(movie_data, data, genre_list, method_num, 
                         x_lim=[-1.5,3], y_lim=[-2,3], save_plot=False,
                         save_folder='figures/', ext='.png'):
    """
    Plots covariance matrix ellipses of listed genres on one plot.
    """
    cmap = cm.get_cmap("viridis")
    # initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # plot ellipse for each genre
    for i in range(len(genre_list)):
        genre = genre_list[i]
        # get color factor
        c = float(i)/(len(genre_list)-1)
        # plot
        xc, yc, lambda_, eig_vecs, var = get_cov_eig(movie_data, data, genre)
        label = "%s, det(cov)=%.3f" % (genre, np.sqrt(var))
        plot_cov_ellipse(xc, yc, lambda_, eig_vecs, ax=ax, label=label, color=cmap(c))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title("Covariance Ellipses of Genres with Method %i" %(method_num),
                 fontsize=18)
    plt.legend(loc="best", fontsize=12)
    
    if save_plot:
        save_name = "method_%i" % method_num
        for i in range(len(genre_list)):
            save_name += "_" + genre
        plt.savefig(save_folder + save_name + ext, bbox_inches="tight")

# main function for running methods
if __name__=='__main__':
    # load projected data (2 x M and 2 x N)
    users_1, users_2, users_3, movies_1, movies_2, movies_3 = load_projections()
    
    # load movies and ratings
    movie_data, rating_data = load_data()
    
    data_all = [movies_1, movies_2, movies_3]
    sorted_genre_lists, genre_vars = order_genres_by_variance(movie_data, data_all)
    
    for i in range(len(data_all)):
        data = data_all[i]
        g = sorted_genre_lists[i]
        method_num = i+1
        sparse_and_compact_genres = [g[0], g[1], g[-3], g[-2]]
        plot_genre_ellipses(movie_data, data, sparse_and_compact_genres,
                            method_num, save_plot=True)
        
    data_all = [users_1, users_2, users_3]
    for i in range(len(data_all)):
        data = data_all[i]
        g = sorted_genre_lists[i]
        method_num = i+1
        sparse_and_compact_genres = [g[0], g[1], g[-3], g[-2]]
        plot_genre_ellipses(movie_data, data, sparse_and_compact_genres,
                            method_num, save_plot=True)
