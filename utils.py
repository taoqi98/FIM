import numpy as np
from sklearn.metrics import roc_auc_score

import time
import json
def dump_result(appraoch,result,path = './result.txt'):
    timestr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    result = json.dumps(result)
    appraoch = json.dumps(appraoch)
    print(timestr)
    print(result)
    print(appraoch)
    with open(path,'a') as f:
        f.write(appraoch+'\n')
        f.write(timestr+'\n')
        f.write(result+'\n')
        f.write('\n')

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def evaluate(user_scorings,news_scorings,Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(Impressions)):
        docids = Impressions[i][0]
        labels = Impressions[i][1]
        uv = user_scorings[i]
        
        nv = news_scorings[docids]
        score = np.dot(nv,uv)
        

        auc = roc_auc_score(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    return AUC, MRR, nDCG5, nDCG10