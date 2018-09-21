import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/home/pratik/movieRec/ml-100k/u.data', sep='\t', names=header)

#print (df.head())


from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]



train_data_matrix = np.zeros((n_users, n_items))

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances


def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


user_similarity = fast_similarity(train_data_matrix, kind='user')
item_similarity = fast_similarity(train_data_matrix, kind='item')
print (item_similarity[:4, :4])


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
	    
	    
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred

#pred = predict_topk(train_data_matrix, user_similarity, kind='user', k=40)

pred = predict_topk(train_data_matrix, item_similarity, kind='item', k=40)
print (pred)

idx_to_movie = {}
with open("/home/pratik/movieRec/ml-100k/u.item", encoding = "ISO-8859-1") as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[1]
        
def top_k_movies(similarity, mapper, movie_idx, k=6):
    return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

idx = 7 # Movie Id
movies = top_k_movies(item_similarity, idx_to_movie, idx)
#posters = tuple(Image(url=get_poster(movie, base_url)) for movie in movies)
print (movies)



