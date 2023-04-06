import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd, numpy as np
from torch.utils.data import DataLoader
import json
import glob
import os

# e.g path = Session 1, Ses01F_impro01
def read_dialogues(path_to_sentences, path_to_labels):
    def find_id_label(id):
        with open(path_to_labels) as file:
            annotations = [line.rstrip() for line in file if str(id) in line]
        if (len(annotations) < 1):
            return "oth"
        
        for label in labels_gt:
            if label in annotations[0]:
                if label == "xxx":
                    return "oth"
                else:
                    return label
        print(annotations[0])
        assert False
        
    # sentences
    import sys
    print(sys.path)
    with open(path_to_sentences) as file:
        sentences = [line.rstrip() for line in file if line[:3]=="Ses"]
    
    labels_gt = {'hap', 'sad', 'neu', 'ang', 'exc', 'fru', 'sur','fea','dis','oth','xxx'}
    labels= {"Other":"oth","Disgust":"dis","Fear":"fear","Neutral":"neu","Frustration":"fru","Anger":"ang","Sadness":"sad","Excited":"exc","Happiness":"hap","Other":"oth","Surprise":"sur"}
            
    print("number of sentences", len(sentences))
    
    dialogue = []
    for sentence in sentences:
        text = sentence.split(":",1)[1] # take only sentence after :
        id_to_label = sentence.split(" ")[0]
        speaker = id_to_label[-4] # M or F
        emotion = find_id_label(id_to_label)
        sentiment = "None"
        act = "None"
        result = {"emotion":emotion,"speaker":speaker,"text":text}
        dialogue.append(result)
    
    return dialogue

all_sessions = ["Session1","Session2","Session3","Session4","Session5"]
for session in all_sessions:
    path_to_sentence = "/home/zhiliang/Downloads/IEMOCAP_full_release/IEMOCAP_full_release/"+session+"/dialog/transcriptions"
    path_to_label = "/home/zhiliang/Downloads/IEMOCAP_full_release/IEMOCAP_full_release/"+session+"/dialog/EmoEvaluation/"
    
    os.chdir(path_to_sentence)
    sentence_files = glob.glob('*.txt')
    os.chdir(path_to_label)
    label_files = glob.glob('*.txt')
    assert len(sentence_files) == len(label_files)
    for s, l in zip(sentence_files, label_files):
        d = read_dialogues("/home/zhiliang/Downloads/IEMOCAP_full_release/IEMOCAP_full_release/"+session+"/dialog/transcriptions/"+s,"/home/zhiliang/Downloads/IEMOCAP_full_release/IEMOCAP_full_release/"+session+"/dialog/EmoEvaluation/"+l)
        p = np.random.uniform(0,1)
        os.chdir("/home/zhiliang/dev/GNN_ERC/baseline/DialogueGCN")
        if p < 0.6:
            with open('IEMOCAP_features/train.json', 'a') as the_file:
                to_write = {"fold":"train","topic":"","dialogue":d}
                the_file.write(json.dumps(to_write)+'\n')
        elif p < 0.8:
            with open('IEMOCAP_features/test.json', 'a') as the_file:
                to_write = {"fold":"test","topic":"","dialogue":d}
                the_file.write(json.dumps(to_write)+'\n')
        elif p < 1:
            with open('IEMOCAP_features/valid.json', 'a') as the_file:
                to_write = {"fold":"valid","topic":"","dialogue":d}
                the_file.write(json.dumps(to_write)+'\n')