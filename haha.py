import numpy as np  
import random
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate(P, Q, train_set, test_set, K=5):
    recall_sum = 0
    precision_sum = 0
    n_users = 200948

    for u in range(n_users):

        score = np.dot(Q, P[u]) #사용자 u의 모든 아이템 점수
        train_items = train_set[train_set[:, 0] == u, 1] 
        if len(test_items) == 0:
            continue
        score[train_items] = -np.inf     #이미 본 아이템 제외
        top_K = np.argpartition(score, -K)[-K:]    #상위 K개
        test_items = test_set[test_set[:, 0] == u, 1] #사용자 u의 테스트 아이템
        hit = np.intersect1d(top_K, test_items)
        recall = len(hit) / len(test_items)
        precision = len(hit) / K
        recall_sum += recall
        precision_sum += precision

    avg_recall = recall_sum / n_users
    avg_precision = precision_sum / n_users
    print("Recall@", K, ":", avg_recall)
    print("Precision@", K, ":", avg_precision)

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

user_to_idx = {v: i for i, v in enumerate(ratings["userId"].unique())}
movie_to_idx = {v: i for i, v in enumerate(movies["movieId"].unique())}

train_set = pd.DataFrame({
    "userId": ratings["userId"].map(user_to_idx),
    "movieId": ratings["movieId"].map(movie_to_idx)
})

all_data = train_set.copy()
test_set = train_set.groupby("userId").tail(2)      #사용자마다 끝에 두개를 테스트로
train_set = train_set.drop(test_set.index)          #테스트제외

item = np.arange(len(movie_to_idx))           #아이템배열
train_set = train_set.to_numpy().copy()       #넘파이배열로 변경
test_set = test_set.to_numpy()

n_epoch = 20
n_users = 200948         
n_items = 87585          
k = 5

P = np.random.normal(0, 0.01, (n_users, k))   #무작위 사용자 벡터
Q = np.random.normal(0, 0.01, (n_items, k))   #무작위 아이템 벡터

lr = 0.1;         reg = 0.1     

#로스계산
prev_loss = None
count = 0

user_items = all_data.groupby("userId")["movieId"].apply(set).to_dict()

for epoch in range(n_epoch):
    np.random.shuffle(train_set) #섞기
    total_loss = 0
    for  (u, i) in train_set:
        while True:    #미관측 아이템 뽑기
            j = random.choice(item)
            if j in user_items[u]: #중복검사
                continue
            else:
                break
        
        #x_ui, x_uj, x_uij
        x_ui = np.dot(P[u], Q[i])
        x_uj = np.dot(P[u], Q[j])
        x_uij = x_ui - x_uj
        sig = sigmoid(-x_uij)

        p = P[u].copy()
        P[u] = P[u] + lr * (sig * (Q[i] - Q[j]) - reg * P[u])      #wu에 대해 미분 = hi-hj
        Q[i] = Q[i] + lr * (sig * p - reg * Q[i])                  #hi에 대해 미분 = wu
        Q[j] = Q[j] + lr * (sig * -p - reg * Q[j])                 #hj에 대해 미분 = -wu
        
        loss = -np.log(sigmoid(x_uij))
        total_loss += loss

    avg_loss = total_loss / len(train_set)

    #멈추기 조건
    if prev_loss is not None:
        if abs(prev_loss - avg_loss) < 1e-4:
            count += 1
        else:
            count = 0

        if count >= 5:
            break
    prev_loss = avg_loss
    evaluate(P, Q, train_set, test_set, K = 10)   #평가출력






























