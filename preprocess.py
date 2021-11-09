import os
from collections import Counter
import re
import json
import numpy as np

MAX_TITLE_LEN = 30
MAX_SNI_LEN=100
MAX_VERT_NUM=3
MAX_ENTITY_NUM = 10



def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def read_news(path,attr,filenames,filer_num=3):
    news={}
    category=[]
    news_index={}
    index=1
    word_cnt=Counter()

    entity_dict = {}
    entity_index = 1
    
    entity_satorid = {}
    
    with open(os.path.join(path,filenames)) as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id,vert,title,snipplet, _, _, _, entity= splited[0:8]
        news_index[doc_id]=index
        index+=1
        
        if 'vert' in attr:
            vert = [v for v in vert.split(',') if v.startswith('hp1')]
            category.extend(vert)
        else:
            vert = None
        title = title.lower()
        title=word_tokenize(title)
        if 'sni' in attr:
            snipplet = snipplet.lower()
            snipplet=word_tokenize(snipplet)
        else:
            snipplet = []
        
        entities = []
        if 'entity' in attr:
            entity = json.loads(entity)
            for e in entity:
                label = e['Label']
                label = e['SatoriId']
                if not label in entity_dict:
                    entity_dict[label] = entity_index
                    entity_index += 1
                entities.append(label)
        else:
            entities = None
        
        news[doc_id]=[vert,title,snipplet,entities]     
        word_cnt.update(snipplet+title)
                
    word = [k for k , v in word_cnt.items() if v > filer_num]
    word_dict = {k:v for k, v in zip(word, range(1,len(word)+1))}
    category=list(set(category))
    category_dict={}
    index=1
    for c in category:
        category_dict[c]=index
        index+=1

    return news,news_index,category_dict,word_dict,entity_dict,





def get_doc_input(attr,news,news_index,category,word_dict,entity_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_TITLE_LEN),dtype='int32')
    if 'sni' in attr:
        news_sni=np.zeros((news_num, MAX_SNI_LEN),dtype='int32')
    else:
        news_sni = None
    if 'vert' in attr:
        news_vert=np.zeros((news_num,MAX_VERT_NUM),dtype='int32')
    else:
        news_vert = None
    if 'entity' in attr:
        news_entity = np.zeros((news_num,MAX_ENTITY_NUM),dtype='int32')
    else:
        news_entity = None
        
    for key in news:    
        vert,title,snipplet,entity=news[key]
        doc_index=news_index[key]
        if 'vert' in attr:
            for vert_id in range(min(MAX_VERT_NUM,len(vert))):
                news_vert[doc_index,vert_id] = category[vert[vert_id]]
                
        for word_id in range(min(MAX_TITLE_LEN,len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index,word_id]=word_dict[title[word_id].lower()]
                
        if 'sni' in attr:
            for word_sid in range(min(MAX_SNI_LEN,len(snipplet))):
                if snipplet[word_sid] in word_dict:
                    news_sni[doc_index, word_sid]=word_dict[snipplet[word_sid].lower()]
                    
        if 'entity' in attr:
            for ei in range(min(MAX_ENTITY_NUM,len(entity))):
                e = entity[ei]
                if e in entity_dict:
                    news_entity[doc_index,ei] = entity_dict[e]
        
    return news_title, news_vert, news_sni, news_entity

def parse_train_user(news_index,root_path,filename):
    with open(os.path.join(root_path,filename)) as f:
        s = f.read()
    user = json.loads(s)
    
    TrainUsers = {}
    UserId2Index = {}
    UserIndex2Id = {}
    index = 1

    for userid in user:
        UserId2Index[userid] = index
        UserIndex2Id[index] = userid
        index += 1
        TrainUsers[userid] = [[],[]]
        info = user[userid]
        for i in range(len(info)):
            if news_index[info[i][0]] in TrainUsers[userid][0]:
                continue
            TrainUsers[userid][0].append(news_index[info[i][0]])
            TrainUsers[userid][1].append(info[i][1])
        TrainUsers[userid][1] = np.array(TrainUsers[userid][1])
    return TrainUsers,UserId2Index,UserIndex2Id


def parse_train_samples(news_index,Train_UserId2Index,root_path,filename = 'train.json',):
    with open(os.path.join(root_path,filename)) as f:
        s = f.read()
        samples = json.loads(s)
    
    imp_docids = []
    imp_labels = []
    imp_userindex = []
    imp_time = []
    
    cnt = 0
    for userid in samples:
        sample = samples[userid]
        uindex = Train_UserId2Index[userid]
        
        for i in range(len(sample)):
            pid,nid,eventime = sample[i]
            pid = news_index[pid]
            nid = news_index[nid]
            imp_docids.append([pid,nid])
            imp_time.append(eventime)
            imp_userindex.append(uindex)
            imp_labels.append([1,0])
        cnt += 1 

    imp_docids = np.array(imp_docids)
    imp_labels = np.array(imp_labels)
    imp_userindex = np.array(imp_userindex,dtype='int32')
    imp_time = np.array(imp_time,dtype='int32')
    
    return imp_docids, imp_labels, imp_userindex,imp_time

def parse_test_impression(news_index,root_path,filename='test.json',max_clicked_news=50):
    with open(os.path.join(root_path,filename)) as f:
        s = f.read()
        samples = json.loads(s)
    
    impressions = []
    users = []
    g1 = 0
    g2 = 0
    for sid in range(len(samples)):
        pos,neg,imp_time,click_news,click_time = samples[sid]

        docids = pos+neg
        labels = [1]*len(pos) + [0]*len(neg)
        for i in range(len(docids)):
            if not docids[i] in news_index:
                docids[i] = 0
                g1 += 1
            else:
                docids[i] = news_index[docids[i]]
            g2 += 1

        for i in range(len(click_news)):
            if not click_news[i] in news_index:
                click_news[i] = 0
            else:
                click_news[i] = news_index[click_news[i]]
        if len(click_news)>max_clicked_news:
            click_news = click_news[-max_clicked_news:]
        else:
            click_news = [0]*(max_clicked_news-len(click_news))+click_news
        docids = np.array(docids)
        labels = np.array(labels)
        click_news = np.array(click_news)
        impressions.append([docids,labels,i])
        users.append(click_news)
    users = np.array(users)
    print(g1,g2,g1/g2)
    return impressions,users

def parse_test_impression(news_index,root_path,filename='test.json'):
    with open(os.path.join(root_path,filename)) as f:
        s = f.read()
        samples = json.loads(s)
    
    impressions = []
    g1 = 0
    g2 = 0
    for sid in range(len(samples)):
        pos,neg,imp_time,uid = samples[sid]

        docids = pos+neg
        labels = [1]*len(pos) + [0]*len(neg)
        for i in range(len(docids)):
            if not docids[i] in news_index:
                docids[i] = 0
                g1 += 1
            else:
                docids[i] = news_index[docids[i]]
            g2 += 1

        docids = np.array(docids)
        labels = np.array(labels)
        impressions.append([docids,labels,uid,imp_time])
    return impressions

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word


def load_entity_matrix(root_data_path,entity_dict,entity2satorid):
    
    with open(os.path.join(root_data_path,'entity','entity2id.txt')) as f:
        lines = f.readlines()

    satorid2line = {}

    for i in range(1,len(lines)):
        sid, index = lines[i].strip('\n').split('\t')
        index = int(index)
        satorid2line[sid] = index
        
    with open(os.path.join(root_data_path,'entity','entity2vec.vec')) as f:
        lines = f.readlines()
        
    stat = 0
    entity_matrix = np.zeros((len(entity2satorid)+1,100))
    for e in entity2satorid:
        sid = entity2satorid[e]
        if not sid in satorid2line:
            stat += 1
            continue
        eindex = entity_dict[e]
        index = satorid2line[sid]
        line = lines[index]
        line = line.strip('\n').split('\t')[:-1]
        for j in range(len(line)):
            line[j] = float(line[j])
            entity_matrix[eindex,j] = line[j]
            
    print(stat/len(entity_dict))
    
    return entity_matrix