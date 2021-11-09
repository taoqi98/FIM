import numpy
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
import keras.layers as layers
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from keras.optimizers import *

from preprocess import *
from models import *


def get_doc_encoder(news_encoder_name,use_relu,length,word_embedding_layer):

    sentence_input = Input(shape=(length,), dtype='int32')
    word_vecs = word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    if news_encoder_name == 'CNN':
        word_rep = Conv1D(400,kernel_size=3)(droped_vecs)
    elif news_encoder_name == 'SelfAtt':
        word_rep = Attention(20,20)([droped_vecs,droped_vecs,droped_vecs])
    if use_relu:
        word_rep = keras.layers.Activation('relu')(word_rep)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = AttentivePooling(length,400)(droped_rep)
    sentEncodert = Model(sentence_input, title_vec)
    return sentEncodert

def get_vert_encoder(category_dict):
    input_vert = keras.Input(shape=(MAX_VERT_NUM,), dtype="int32")
    vert_embedding = layers.Embedding(
        len(category_dict)+1, 400, trainable=True
    )

    vert_emb = vert_embedding(input_vert)
    vert_emb = AttentivePooling(MAX_VERT_NUM,400)(vert_emb)
    #pred_vert = Dense(400)(vert_emb)
    pred_vert = Reshape((1, 400))(vert_emb)

    model = keras.Model(input_vert, pred_vert, name="vert_encoder")
    return model

def get_entity_encoder(length,entity_embedding_layer):

    entity_input = Input(shape=(length,), dtype='int32')
    entity_vecs = entity_embedding_layer(entity_input)
    droped_vecs = Dropout(0.2)(entity_vecs)
    
    entity_rep = Attention(20,20)([droped_vecs,droped_vecs,droped_vecs])

    droped_rep = Dropout(0.2)(entity_rep)
    entity_vec = AttentivePooling(length,400)(droped_rep)
    sentEncodert = Model(entity_input, entity_vec)
    return sentEncodert

def get_news_encoder(word_embedding_matrix,entity_dict,category_dict,news_encoder_name, use_relu, attr=['title', 'vert', 'sni']):
    
    if 'title' in attr or 'sni' in attr:
        word_embedding_layer= Embedding(word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], weights=[word_embedding_matrix],trainable=True)
    if 'entity' in attr:
        entity_embedding_layer = Embedding(len(entity_dict)+1,300,trainable=True)

    ONE_FEATURE_LENGTH = {'title':MAX_TITLE_LEN,'entity':MAX_ENTITY_NUM,'sni':MAX_SNI_LEN,'vert':MAX_VERT_NUM}
        
    MAX_LENGTH= 0
    bias = 0
    FeatureTable = {}
    for a in attr:
        MAX_LENGTH = bias + ONE_FEATURE_LENGTH[a]
        FeatureTable[a] = [bias,MAX_LENGTH]
        bias = MAX_LENGTH
    print(FeatureTable)
    
    input_feature = keras.Input(shape=(MAX_LENGTH,), dtype="int32")
    
    if 'title' in attr:
        title_encoder = get_doc_encoder(news_encoder_name,use_relu,MAX_TITLE_LEN,word_embedding_layer)
    else:
        title_encoder = None
    if 'sni' in attr:
        sni_encoder = get_doc_encoder(news_encoder_name,use_relu,MAX_SNI_LEN,word_embedding_layer)
    else:
        sni_encoder = None
    if 'vert' in attr:
        vert_encoder = get_vert_encoder(category_dict)
    else:
        vert_encoder = None
    if 'entity' in attr:
        entity_encoder  = get_entity_encoder(MAX_ENTITY_NUM,entity_embedding_layer)
    else:
        entity_encoder = None

    EncoderTable = {'title':title_encoder,'sni':sni_encoder,'vert':vert_encoder,'entity':entity_encoder}

    feature = []
    for a in attr:
        start,ed = FeatureTable[a]
        attr_feature = layers.Lambda(lambda x: x[:,start:ed])(input_feature)
        attr_encoder = EncoderTable[a]
        attr_rep = attr_encoder(attr_feature)
        if len(attr) > 1:
            attr_rep = keras.layers.Reshape((1,400))(attr_rep)
        feature.append(attr_rep)
    
    if len(feature)>1:
        feature = keras.layers.Concatenate(axis=1)(feature)
        news_rep = AttentivePooling(len(attr), 400)(feature)
    else:
        news_rep = feature[0]
    
    model = keras.Model(input_feature, news_rep, name="news_encoder")
    return model, MAX_LENGTH


def create_model(news_encoder_name,use_relu,user_encoder_name,word_dict,category_dict,entity_dict,title_word_embedding_matrix,attr,npratio = 1):

    news_encoder, MAX_LENGTH = get_news_encoder(title_word_embedding_matrix,entity_dict,category_dict,news_encoder_name,use_relu,attr)    
    clicked_title_input = Input(shape=(50,MAX_LENGTH), dtype='int32') #(batch_size, 50, 30)
    user_vecs = TimeDistributed(news_encoder)(clicked_title_input) #(batch_size, 50, 400)

    dim = 400
    
    if user_encoder_name == 'Att':
        user_vec = AttentivePooling(50,dim)(user_vecs) 
    elif user_encoder_name == 'SelfAtt':
        user_vecs = Attention(20,20)([user_vecs,user_vecs,user_vecs]) #(batch_size, 50, 400)
        user_vec = AttentivePooling(50,dim)(user_vecs) # #(batch_size, 400)
    elif user_encoder_name == 'GRU':
        user_vec = GRU(400)(user_vecs)
    elif user_encoder_name == 'LSTUR':
        user_vecs1 = Attention(20,20)([user_vecs,user_vecs,user_vecs]) #(batch_size, 50, 400)
        user_vec1 = AttentivePooling(50,400)(user_vecs1) # #(batch_size, 400)
        user_vec2 = GRU(400)(user_vecs)
        user_vec = keras.layers.Concatenate(axis=-1)([user_vec1,user_vec2])
        user_vec = keras.layers.Dense(400)(user_vec)
        
    title_inputs = Input(shape=(1+npratio,MAX_LENGTH),dtype='int32')

    # News
 
    news_vecs = TimeDistributed(news_encoder)(title_inputs) # (batch_size,2,400)
        
    scores = keras.layers.Dot(axes=-1)([user_vec,news_vecs]) #(batch_size,1+1,) 
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     
    
    model = Model([clicked_title_input,title_inputs,],logits) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001), 
                  metrics=['acc'])
    
    user_encoder = Model([clicked_title_input],user_vec)
    
    
    return model,news_encoder, user_encoder,
