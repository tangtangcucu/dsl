import numpy as np  
import random
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

pairs = np.array([[0, 0],     #ABCD
                 [0, 1],
                 [0, 4],
                 [1, 0],
                 [1, 2],
                 [2, 1],
                 [2, 4],
                 [3, 3]])
item = np.array([0, 1, 2, 3, 4])   #abcde

n_epoch = 100
n_users = 4
n_items = 5
k = 3

P = np.random.random((n_users, k))   #무작위 사용자 벡터
Q = np.random.random((n_items, k))   #무작위 아이템 벡터
np.random.shuffle(pairs)

lr = 0.1      #학습률
reg = 0.1    #정규화세기

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

        

print(np.dot(P[3], Q[3]))  #관측
print(np.dot(P[3], Q[2]))  #미관측
u = 0
count = (pairs[:, 0] == u).sum()

print(count)