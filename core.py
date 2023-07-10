import pickle
import numpy as np 
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

addr = './'

user_recomms_file = open(addr + 'user_recomms.pkl', "rb")
user_recomms = pickle.load(user_recomms_file)
user_recomms_file.close()


model.load_state_dict(torch.load(addr + 'model.pth'))

final_movies_file = open(addr + 'final_movies.pkl', "rb")
final_movies = pickle.load(final_movies_file)
final_movies_file.close()

movie_embeds_file = open(addr + 'movie_embeds.pkl', "rb")
movie_embeds = pickle.load(movie_embeds_file)
movie_embeds_file.close()

btree_file = open(addr + 'btree.pkl', "rb")
btree = pickle.load(btree_file)
btree_file.close()

user_embeds_file = open(addr + 'user_embeds.pkl', "rb")
user_embeds = pickle.load(user_embeds_file)
user_embeds_file.close()

user_mapping_file = open(addr + 'user_mapping.pkl', "rb")
user_mapping = pickle.load(user_mapping_file)
user_mapping_file.close()

movie_mapping_file = open(addr + 'movie_mapping.pkl', "rb")
movie_mapping = pickle.load(movie_mapping_file)
movie_mapping_file.close()

user_pos_items_file = open(addr + 'user_pos_items.pkl', "rb")
user_pos_items = pickle.load(user_pos_items_file)
user_pos_items_file.close()

def create_user_embedding(movie_ratings, movies_df):
    # Convert the movie_ratings dictionary to a dataframe
    user_ratings_df = pd.DataFrame.from_dict(movie_ratings, orient='index', columns=['rating'])
    user_ratings_df['movieId'] = user_ratings_df.index
    # Merge the user_ratings_df with the movies_df to get the movie embeddings
    user_movie_embeddings = user_ratings_df.merge(movies_df, on='movieId', how='left')
    
    print(user_ratings_df)
    print(user_movie_embeddings)

    # Multiply the ratings with the movie embeddings
    user_movie_embeddings = user_movie_embeddings.iloc[:, 2:].values * user_movie_embeddings['rating'].values[:, np.newaxis]

    # Calculate the user embedding as the sum of the movie embeddings
    user_embedding = np.sum(user_movie_embeddings, axis=0)
    np.nan_to_num(user_embedding, 0)
    print(user_movie_embeddings.shape)
    return user_embedding

def find_closest_user(user_embedding, tree, user_embeddings):
    # Query the BallTree to find the closest user to the given user_embedding
    _, closest_user_index = tree.query([user_embedding], k=1)

    # Get the embedding of the closest user
    closest_user_embedding = user_embeddings.iloc[closest_user_index[0][0]]

    return closest_user_embedding


def output_list(movie_ratings, movies_df = movie_embeds, tree = btree, user_embeddings = user_embeds, movies = final_movies):
    user_embed = create_user_embedding(movie_ratings, movie_embeds)
    # Call the find_closest_user function with the pre-built BallTree
    closest_user_embed = find_closest_user(user_embed, tree, user_embeds)
    recomms = user_recomms[int(closest_user_embed['userId'])]
    out = [movies['title'].iloc[movie_id] for movie_id in recomms]
    return out

# output_list({1:1,2:2,3:3,4:4,5:5})

