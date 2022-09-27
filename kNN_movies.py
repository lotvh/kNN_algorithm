#!/usr/bin/env python3

import numpy as np
from kNN_algorithm import kNN_algorithm

def recommend_movies(movie_query, k):
    with open('movies.txt', 'r') as md:
        raw_movies_data = [line.strip().split(',') for line in md.readlines()][1:]

    # preprocess data to input algorithm
    movies_data = [ [ list(map(float, row[2:])) , row[0] ] for row in raw_movies_data ]

    # use algorithm to return IDs of recommended movies
    movie_IDs = kNN_algorithm(movies_data, movie_query, k, data_type='raw')

    # find names of recommended movies
    movie_names = [movie_entry[1] for movie_entry in raw_movies_data if str(movie_entry[0]) in movie_IDs]

    return movie_names

if __name__ == '__main__':
    # example to show how to use the algorithm
    movie_example = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]
    recommended_movies = recommend_movies(movie_example, 5)

    print('The recommended movies are: ', ', '.join([str(movie) for movie in recommended_movies]))
