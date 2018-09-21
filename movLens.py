import pandas as pd
import numpy as np 
import random

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/home/pratik/movieRec/ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('/home/pratik/movieRec/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)
'''
print (users.head())
print (users.shape)
print (ratings.head())
print (ratings.shape)
print (movie_ratings.head())
print (movie_ratings.shape)
print (lens.head())
print (lens.shape)
'''
movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
#print (movie_stats.head())

atleast_100 = movie_stats['rating']['size'] >= 100
topRated = movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:5]
#print (topRated)

context = topRated.to_dict('index')
print ("##################################################")
print (context)

for i in context:
	print (i)
	print (context[i][('rating','mean')])

{% if topList %}
    <ul>
    {% for i in topList %}
        
        {{i}}
    {% endfor %}
    </ul>
    
{% else %}
    <p>No polls are available.</p>
{% endif %}


with open('/home/pratik/movieRec/ml-100k/u.item') as f:
        reader = csv.reader(f)
        for row in reader:
            _, created = Teacher.objects.get_or_create(
                first_name=row[0],
                last_name=row[1],
                middle_name=row[2],
                )

