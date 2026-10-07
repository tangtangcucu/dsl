#Impliment the BPR - 동아대학교 컴퓨터공학과 김미리내

import numpy as np  
import random
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluate(P, Q, n_users, train_set, test_set, K=10):
    
    recall_sum = 0
    precision_sum = 0
    n_users = n_users

    for u in range(n_users):
        
        score = np.dot(Q, P[u])                            #사용자 u의 모든 아이템 점수
        train_items = train_set[train_set[:, 0] == u, 1]
        score[train_items] = -np.inf                       #이미 본 아이템 제외
        test_items = test_set[test_set[:, 0] == u, 1]      #사용자 u의 테스트 아이템

        top_K = np.argpartition(score, -K)[-K:]    #상위 K개
        hit = np.intersect1d(top_K, test_items)
        recall = len(hit) / len(test_items)
        precision = len(hit) / K
        recall_sum += recall          
        precision_sum += precision     

    avg_recall = recall_sum / n_users
    avg_precision = precision_sum / n_users
    print(f"Recall@{k} : {avg_recall}")          #맞춘 개수/관측템
    print(f"Precision@{k} : {avg_precision}")    #맞춘 개수/추천템
def validation(P, Q, n_users, train_set, vali_set, K=10):
    
    recall_sum = 0
    n_users = n_users

    for u in range(n_users):
        
        score = np.dot(Q, P[u])                            #사용자 u의 모든 아이템 점수
        train_items = train_set[train_set[:, 0] == u, 1]
        score[train_items] = -np.inf                       #이미 본 아이템 제외
        vali_items = vali_set[vali_set[:, 0] == u, 1]      #사용자 u의 테스트 아이템

        top_K = np.argpartition(score, -K)[-K:]    #상위 K개
        hit = np.intersect1d(top_K, vali_items)
        recall = len(hit) / len(vali_items)
        recall_sum += recall          

    avg_recall = recall_sum / n_users
    return avg_recall       #맞춘 개수/관측템
def split_data(data):

    test_idx = data.groupby("userId", group_keys=False).apply(lambda x: x.tail(int(len(x) * 0.1))).index
    vali_idx = data.groupby("userId", group_keys=False).apply(lambda x: x.head(int(len(x) * 0.1))).index

    test_set = data.loc[test_idx].copy()    #트레인에서 뒤에 10%
    vali_set = data.loc[vali_idx].copy()    #트레인에서앞에  10%
    train_set = data.drop(test_idx)         #나머지         80%
    train_set = train_set.drop(vali_idx)
    user_items = train_set.groupby("userId")["movieId"].apply(set).to_dict()


    train_set = train_set.to_numpy().copy()           #넘파이배열로 변경
    vali_set = vali_set.to_numpy()
    test_set = test_set.to_numpy()


    return train_set, vali_set, test_set, user_items

class BPR:
    def __init__(self, P, Q):
        self.P = P
        self.Q = Q
        
    def MF(self, u, i, j):    #x_ui, x_uj, x_uij
        x_ui = np.dot(self.P[u], self.Q[i])
        x_uj = np.dot(self.P[u], self.Q[j])
        return x_ui - x_uj   # == x_uij

    def SGD(self, u, i, j, lr, reg):
        sig = sigmoid(-self.MF(u, i, j))
        p = self.P[u].copy()
        self.P[u] += lr * (sig * (self.Q[i] - self.Q[j]) - reg * self.P[u])     #wu에 대해 미분 = hi-hj
        self.Q[i] += lr * (sig * p - reg * self.Q[i])                           #hi에 대해 미분 = wu
        self.Q[j] += lr * (sig * -p - reg * self.Q[j])                          #hj에 대해 미분 = -wu


if __name__ == "__main__":

    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")

    user_to_idx = {v: i for i, v in enumerate(ratings["userId"].unique())}
    movie_to_idx = {v: i for i, v in enumerate(movies["movieId"].unique())}

    data = pd.DataFrame({
        "userId": ratings["userId"].map(user_to_idx),
        "movieId": ratings["movieId"].map(movie_to_idx)
    })

    train_set, vali_set, test_set, user_items = split_data(data)

    n_epoch = 50
    n_users = len(user_to_idx)    
    n_items = len(movie_to_idx)    
    k = 50

    P = np.random.normal(0, 0.01, (n_users, k))   #무작위 사용자 벡터
    Q = np.random.normal(0, 0.01, (n_items, k))   #무작위 아이템 벡터

    lr = 0.05;         reg = 0.0025   


    bpr = BPR(P, Q)  #클래스 객체 생성

    best_vali = -1
    count = 0

    for epoch in range(n_epoch):
        np.random.shuffle(train_set) #섞기
        total_loss = 0
        for  (u, i) in train_set:
            while True:    #미관측 아이템 뽑기
                j = random.randrange(n_items)
                if j in user_items[u]: #중복검사
                    continue
                else:
                    break
        
            bpr.SGD(u, i, j, lr, reg)    #경사하강법

            x_uij = bpr.MF(u, i, j)
            loss = -np.log(sigmoid(x_uij))   
            total_loss += loss

        avg_loss = total_loss / len(train_set)
        vali = validation(P, Q, n_users, train_set, vali_set, K = 10)   #매 epoch마다 검증

        #멈추기 조건 : 최고 recall 갱신못하면 종료
        if vali > best_vali:  
            best_vali = vali
            best_P = P.copy()
            best_Q = Q.copy()

            count = 0
        else:
            count += 1
        if count > 4:
            break

    evaluate(best_P, best_Q, n_users, train_set, test_set, K = 10)   #평가출력
    print(f"loss : {avg_loss}")



##감사합니다!!!!!!!!!