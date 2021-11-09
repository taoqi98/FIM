import numpy as np

class NewsFetcher():
    def __init__(self,attr,news_title, news_vert, news_sni,news_entity):
        self.title = news_title
        self.sni = news_sni
        self.vert = news_vert
        self.entity = news_entity
        
        self.attr = attr
    
    def fetch_news(self,docids):
        attr = self.attr
        if 'title' in attr:
            title = self.title[docids]
        else:
            title = None
        if 'vert' in attr:
            vert = self.vert[docids]
        else:
            vert = None
        if 'sni' in attr:
            sni = self.sni[docids]
        else:
            sni = None
        if 'entity' in attr:
            entity = self.entity[docids]
        else:
            entity = None
        Table = {'title':title, 'vert':vert, 'sni':sni, 'entity':entity}
        
        feature = []
        
        for a in attr:
            feature.append(Table[a])
        if len(feature) == 1:
            feature = feature[0]
        else:
            feature = np.concatenate(feature,axis=-1)
            
 
        return [feature]

from keras.utils import Sequence

class get_hir_train_generator(Sequence):
    def __init__(self,news_fetcher,user_info,UserIndex2Id, news_id,user_id,imp_time, label, batch_size):
        
        self.news_fetcher = news_fetcher
        
        self.user_info = user_info
        self.UserIndex2Id = UserIndex2Id
        
        self.doc_id = news_id
        self.user_id = user_id
        self.imp_time = imp_time
        self.label = label
        
        self.batch_size = batch_size
        self.Imp = label.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.Imp/self.batch_size))

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed >self.Imp:
            ed = self.Imp
        
        doc_ids = self.doc_id[start:ed]
        news_feature = self.news_fetcher.fetch_news(doc_ids)
        
        clicked_ids = []
        for i in range(start,ed):
            uindex = self.user_id[i]
            uid = self.UserIndex2Id[uindex]
            user_click, user_eventime = self.user_info[uid]
            eventime = self.imp_time[i]
            uc = user_click.copy()
            ct = (user_eventime<eventime).sum()
            if ct > 50:
                uc = uc[ct-50:ct]
            else:
                uc = [0]*(50-ct) + uc[:ct]
            clicked_ids.append(uc)
        clicked_ids = np.array(clicked_ids)
        user_feature = self.news_fetcher.fetch_news(clicked_ids)
        
        label = self.label[start:ed]
                
        return (user_feature+news_feature,[label])

class get_hir_user_generator(Sequence):
    def __init__(self,news_fetcher, user_info, imps,batch_size):
        
        self.news_fetcher = news_fetcher
        self.user_info = user_info
        
        self.imps = imps
        
        self.batch_size = batch_size
        self.ImpNum = len(self.imps)
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
            
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        clicked_ids = []
        for i in range(start,ed):
            uid = self.imps[i][2]
            eventime = self.imps[i][3]

            user_click, user_eventime = self.user_info[uid]
            uc = user_click.copy()
            ct = (user_eventime<eventime).sum()
            if ct > 50:
                uc = uc[ct-50:ct]
            else:
                uc = [0]*(50-ct) + uc[:ct]
            clicked_ids.append(uc)
            
        clicked_ids = np.array(clicked_ids)
        user_feature = self.news_fetcher.fetch_news(clicked_ids)
        
        if len(user_feature)>1:
            user_feature = np.concatenate(user_feature, axis=-1)

        return user_feature

class get_hir_news_generator(Sequence):
    def __init__(self,news_fetcher,news_title,batch_size):
        self.title = news_title
        self.news_fetcher = news_fetcher

        self.batch_size = batch_size
        
        self.ImpNum = self.title.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        docids = np.array([i for i in range(start,ed)])
        
        news_feature = self.news_fetcher.fetch_news(docids)
        if len(news_feature)>1:
            news_feature = np.concatenate(news_feature, axis=-1)        
        return news_feature