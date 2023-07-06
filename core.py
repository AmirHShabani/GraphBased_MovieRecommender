import pickle
import numpy as np 
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from sklearn.neighbors import BallTree
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, diffusion_steps=3, add_self_loops=False):
        super().__init__()
        
        # Number of users and items in the graph
        self.num_users = num_users
        self.num_items = num_items
        
        # Embedding dimension for user and item nodes
        self.embedding_dim = embedding_dim
        
        # Number of diffusion steps (K) for multi-scale diffusion
        self.diffusion_steps = diffusion_steps
        
        # Whether to add self-loops to the adjacency matrix
        self.add_self_loops = add_self_loops
        
        # Initialize embeddings for users and items (E^0)
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        # Initialize embedding weights with a normal distribution (mean=0, std=0.1)
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        # Compute the symmetrically normalized adjacency matrix (A_hat or \tilde{A})
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        # Get initial embeddings E^0 for all nodes (users and items)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
        
        # List to store embeddings at each diffusion step (E^1, E^2, ..., E^K)
        embs = [emb_0]
        
        # Initialize the current embeddings to E^0
        emb_k = emb_0

        # Perform multi-scale diffusion for K steps
        for _ in range(self.diffusion_steps):
            # Propagate embeddings and update emb_k using the normalized adjacency matrix
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            # Save embeddings at each diffusion step for later use
            embs.append(emb_k)

        # Stack all the embeddings along the second dimension (stack E^0, E^1, ..., E^K)
        embs = torch.stack(embs, dim=1)
        
        # Calculate the final embeddings by taking the mean of all diffusion embeddings (E^K)
        emb_final = torch.mean(embs, dim=1)  # E^K

        # Split the final embeddings into user embeddings (e_u^K) and item embeddings (e_i^K)
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])  # Splits into e_u^K and e_i^K

        # Returns the final embeddings for users (e_u^K), initial embeddings for users (e_u^0),
        # final embeddings for items (e_i^K), and initial embeddings for items (e_i^0)
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        # The message function is an identity function, i.e., it returns x_j itself
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # Perform message passing and aggregation using the normalized adjacency matrix (A_hat or \tilde{A})
        return matmul(adj_t, x)


model = LightGCN(671, 9125)

def get_movie_recommendations(user_id, num_recomms):
    # Map the user ID to the corresponding index in the model's user embeddings
    user_index = user_mapping[user_id]

    # Retrieve the user embedding for the specified user
    user_embedding = model.users_emb.weight[user_index]

    # Calculate scores for all items using the user embedding
    scores = model.items_emb.weight @ user_embedding

    # Get the indices of the highest scores, including positive items and additional recommendations
    values, indices = torch.topk(scores, k=len(user_pos_items[user_id]) + num_recomms)

    # Retrieve the recommended movies that the user has already rated highly
    rated_movies = [index.cpu().item() for index in indices if index in user_pos_items[user_id]][:num_recomms]
    rated_movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in rated_movies]

    # Retrieve the suggested movies for the user that they have not rated
    suggested_movies = [index.cpu().item() for index in indices if index not in user_pos_items[user_id]][:num_recomms]
    suggested_movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in suggested_movies]

    return rated_movie_ids, suggested_movie_ids

addr = './'

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
    rated_movie_ids, suggested_movie_ids = get_movie_recommendations(closest_user_embed['userId'], 5)
    out1 = [movie_id for movie_id in set(rated_movie_ids + suggested_movie_ids) if movie_id not in movie_ratings.keys()]
    out2 = [movies['title'][idx] for idx in out1]
    return out2

# output_list({1:1,2:2,3:3,4:4,5:5})

