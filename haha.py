import numpy as np  
import random
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train = pd.read_csv("ratings.csv");     train2 = pd.read_csv("movies.csv")
df= pd.DataFrame();                     df2 = pd.DataFrame()     

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

pairs = np.array(df)            #(u, i)      userId, movieId
item = np.array(df2).ravel()    #j           movieId

n_epoch = 3
n_users = 200948
n_items = 87585
k = 5

P = np.random.random((n_users, k))   #무작위 사용자 벡터
Q = np.random.random((n_items, k))   #무작위 아이템 벡터
np.random.shuffle(pairs)

lr = 0.1      #학습률
reg = 0.1     #정규화세기

for epoch in range(n_epoch):
    for (u, i) in pairs:
        while True:    #미관측 아이템 뽑기
            j = random.choice(item)
            if (i == j):
                continue
            else:
                break
        
        #x_ui, x_uj, x_uij
        x_ui = np.dot(P[u], Q[i])
        x_uj = np.dot(P[u], Q[j])
        x_uij = x_ui - x_uj
        sig = sigmoid(-x_uij)
        P[u] = P[u] + lr * (sig * (Q[i] - Q[j]) - reg * P[u])      #wu에 대해 미분 = hi-hj
        Q[i] = Q[i] + lr * (sig * P[u] - reg * Q[i])               #hi에 대해 미분 = wu
        Q[j] = Q[j] + lr * (sig * -P[u] - reg * Q[j])              #hj에 대해 미분 = -wu

        #멈추기 조건

print(np.dot(P[0], Q[16]))  #관측
print(np.dot(P[0], Q[20]))  #미관측





# #사용자 0번에게 추천해주기
# u = 0
# count = (pairs[:, 0] == u).sum()
# l = np.array([])

# for i in range(count):
#     for j in range(n_items):
#         l[i] = np.dot(P[u], Q[j])























# a = 0.1      #학습률
# reg = 0.1    #정규화세기
# W = np.array([[0.2, 0.5],  #철수 / k = 2                 
#              [0.7, 0.1],   #민수
#              [0.3, 0.8]])  #영희
# random.shuffle(W) 

# H = np.array([[0.8, 0.3],  #기생충
#              [0.1, 0.9]])  #짱구
# random.shuffle(H)


# for s in range(3):
#     k = random.randint(1, len(W))
#     i, j = random.randint(1, H_col)  
#     #x_ui, x_uj, x_uij
#     x_ui = np.dot(W[k], H[i])
#     x_uj = np.dot(W[k], H[j])
#     x_uij = x_ui - x_uj
#     sig = sigmoid(-x_uij)
#     W[0] = W[0] + a * (sig * (H[0] - H[1]) - reg * W[0])     #wu(철수)에 대해 미분 = hi-hj
#     H[0] = H[0] + a * (sig * W[0] - reg * H[0])              #hi(기생충)에 대해 미분 = wu
#     H[1] = H[1] + a * (sig * (-W[0]) - reg * H[1])           #hj(짱구)에 대해 미분 = -wu






















