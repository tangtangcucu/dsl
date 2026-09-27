import numpy as np  
import random
import pandas as pd

train = pd.read_csv("ratings.csv")
train2 = pd.read_csv("movies.csv")
df= pd.DataFrame()
df2= pd.DataFrame()

# df["userId"] = train["userId"]
# df["movieId"] = train["movieId"]
# # df["movieId"] = train["movieId"].apply(lambda x : enumerate(x))
# df2["movieId"] = train2["movieId"]


#df userId --> idx ,  movieId --> idx
user_ids = train["userId"].unique()
user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
df["userId"] = train["userId"].map(user_to_idx)
movie_ids = train2["movieId"].unique()
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
df["movieId"] = train["movieId"].map(movie_to_idx)
#df2 movieId --> idx
movie_to_idx2 = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
df2["movieId"] = train2["movieId"].map(movie_to_idx2)
print(df)
print(df2)


# item = np.array(df2)    #movieId
# print(item)
pairs = np.array(df)    #userId, movieId
# print(pairs)
a = np.array([[1, 2], [2, 3]])

# random.shuffle(a)
# for (u, i) in a:
#     print(u, i)

n_users = 100
n_items = 100
k = random.randint(1, 5)

P = random(n_users, k)
Q = random(n_items, k)

j = random(item)



# git remote add origin https://github.com/tangtangcucu/dsl.git
# git push -u origin main