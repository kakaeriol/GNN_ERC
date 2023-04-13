# import pandas as pd
import sys
sys.path.append("..") 
from sysconf import  conf
import pandas as pd, numpy as np, pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


def preprocess_text(x):
    for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punct, ' ')

    x = ' '.join(x.split())
    x = x.lower()
    return x

def flatten_list(lx):
    return [preprocess_text(ii)  for sublist in lx for ii in sublist]

def load_pretrained_glove(glove_path:str=conf["glove_path"])->dict:
    """ This function is run only one first time,
    everytime later it will reload the output that save 
    in the first time
    """
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    # Put your glove embedding path here
    ## check if dictionary already have
    file_vector = glove_path + '.dict_'
    if os.path.isfile(file_vector):
        glv_vector = pd.read_pickle(file_vector)
        return glv_vector
    f = open(glove_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    pd.to_pickle(glv_vector, file_vector)
    print("Completed loading pretrained GloVe model.")
    return glv_vector

def create_data_token(data, tokenizer, max_num_tokens):
    """
    Input one segment of data - create the data token 
    """
    data_token = []
    max_len = []
    token_raw = []
    for id_conv in range(len(data['conversation'])):
        idata = data['conversation'][id_conv]
        i_token = tokenizer.texts_to_sequences(list(map(preprocess_text, idata)))
        i_pad_sequence = pad_sequences(i_token, maxlen=max_num_tokens, padding='post')
        i_token_len = [len(i) for i in i_token]
        max_len.append(max(i_token_len))
        data_token.append(i_pad_sequence)
        token_raw.append(i_token)
    rs = {'data_token': data_token, 'max_len': max_len, 'token_raw': i_token}
    rs.update(data)
    return rs

class Preprocessing_Data_Token:
    """This process is read from raw data pickle,
        and generate the token file, as well as glove embedding"""
    def __init__(self, raw_pkl_file:str, max_num_tokens=250):
        self.raw_pkl_file = raw_pkl_file
        self.data = pd.read_pickle(raw_pkl_file)
        self.tokenizer = Tokenizer()
        self.max_num_tokens = max_num_tokens
        self.glv_embedding_matrix = []
        
    def processing_token_data_segment(self,output_file=None):
        train = self.data['train']
        test = self.data['test']
        dev = self.data['dev']
        ### fitting tokenize
        self.tokenizer.fit_on_texts(flatten_list(train['conversation'])) # need to store
        max_num_tokens = 250
        out_train = create_data_token(train, self.tokenizer, self.max_num_tokens)
        out_test = create_data_token(test, self.tokenizer, self.max_num_tokens)
        out_dev  = create_data_token(dev, self.tokenizer, self.max_num_tokens)
        rs = {'train': out_train, 'dev': out_dev, 'test': out_test, 'tokenizer':self.tokenizer}
        if output_file:
            pd.to_pickle(rs, output_file)
        return rs
    
    def create_pretrain_embedding(self,output_file=None):
        if len(self.tokenizer.word_index) == 0:
            self.tokenizer.fit_on_texts(flatten_list(self.data['train']['conversation'])) # need to store
        glove_vector = load_pretrained_glove()
        word_vector_length = len(glove_vector['the'])
        word_index = self.tokenizer.word_index
        inv_word_index = {v: k for k, v in word_index.items()}
        num_unique_words = len(word_index)

        glv_embedding_matrix = np.zeros((num_unique_words + 1, word_vector_length))
        if output_file:
            pd.to_pickle({'tokenizer': self.tokenizer, 'embedding': glv_embedding_matrix}, output_file)
        self.glv_embedding_matrix = glv_embedding_matrix