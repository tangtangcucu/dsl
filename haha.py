import numpy as np  
import random
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train = pd.read_csv("ratings.csv")
train2 = pd.read_csv("movies.csv")
df= pd.DataFrame();       df2 = pd.DataFrame()     

df["userId"] = train["userId"]
df["movieId"] = train["movieId"]
df2["movieId"] = train2["movieId"]

pairs = np.array(df)    #(u, i)      userId, movieId
item = np.array(df2)    #j           movieId

n_epoch = 3
n_users = 200948
n_items = 87585
k = 5

P = random(n_users, k)
Q = random(n_items, k)
random.shuffle(pairs)

lr = 0.1      #학습률
reg = 0.1    #정규화세기

for epoch in range(n_epoch):
    for (u, i) in pairs:
        j = random(item)
        #x_ui, x_uj, x_uij
        x_ui = np.dot(P[u], Q[i])
        x_uj = np.dot(P[u], Q[j])
        x_uij = x_ui - x_uj
        sig = sigmoid(-x_uij)
        P[u] = P[u] + lr * (sig * (Q[i] - Q[j]) - reg * P[u])      #wu에 대해 미분 = hi-hj
        Q[i] = Q[i] + lr * (sig * P[u] - reg * Q[i])               #hi에 대해 미분 = wu
        Q[j] = Q[j] + lr * (sig * -P[u] - reg * Q[j])              #hj에 대해 미분 = -wu



a = 0.1      #학습률
reg = 0.1    #정규화세기
W = np.array([[0.2, 0.5],  #철수 / k = 2                 
             [0.7, 0.1],   #민수
             [0.3, 0.8]])  #영희
random.shuffle(W) 

H = np.array([[0.8, 0.3],  #기생충
             [0.1, 0.9]])  #짱구
random.shuffle(H)


for s in range(3):
    k = random.randint(1, len(W))
    i, j = random.randint(1, H_col)  
    #x_ui, x_uj, x_uij
    x_ui = np.dot(W[k], H[i])
    x_uj = np.dot(W[k], H[j])
    x_uij = x_ui - x_uj
    sig = sigmoid(-x_uij)
    W[0] = W[0] + a * (sig * (H[0] - H[1]) - reg * W[0])     #wu(철수)에 대해 미분 = hi-hj
    H[0] = H[0] + a * (sig * W[0] - reg * H[0])              #hi(기생충)에 대해 미분 = wu
    H[1] = H[1] + a * (sig * (-W[0]) - reg * H[1])           #hj(짱구)에 대해 미분 = -wu






















