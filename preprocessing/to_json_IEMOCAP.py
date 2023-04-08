import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd, numpy as np
from torch.utils.data import DataLoader
import json
import glob
import os
import argparse

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

def create_json_file(base_IEMOCAP_link, base_output):
    """
    This function out put the json data for train/test/valid datasets for IEMOCAP full dataraw
    base_IEMOCAP_link="/home/zhiliang/Downloads/IEMOCAP_full_release/IEMOCAP_full_release/"
    """
    
    np.random.seed(1111)
    sentence_path = os.path.join(base_IEMOCAP_link, "{}/dialog/transcriptions")
    label_path = os.path.join(base_IEMOCAP_link, "{}//dialog/EmoEvaluation")
    output_path = os.path.join(base_output, "IEMOCAP_features")
    ## check if folder exist!
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    train_json = os.path.join(output_path, "train.json")
    test_json = os.path.join(output_path, "test.json")
    valid_json = os.path.join(output_path, "valid.json")
    all_sessions = ["Session1","Session2","Session3","Session4","Session5"]
    for session in all_sessions:
        path_to_sentence = sentence_path.format(session)
        path_to_label = label_path.format(session)
        os.chdir(path_to_sentence)
        sentence_files = glob.glob('*.txt')
        os.chdir(path_to_label)
        label_files = glob.glob('*.txt')
        assert len(sentence_files) == len(label_files)
        for s, l in zip(sentence_files, label_files):
            i_sp = os.path.join(sentence_path.format(session), s)
            i_lp = os.path.join(label_path.format(session), l)
            d = read_dialogues(i_sp,i_lp)
            p = np.random.uniform(0,1)
            if p < 0.6:
                with open(train_json, 'a') as the_file:
                    to_write = {"fold":"train","topic":"","dialogue":d}
                    the_file.write(json.dumps(to_write)+'\n')
            elif p < 0.8:
                with open(test_json, 'a') as the_file:
                    to_write = {"fold":"test","topic":"","dialogue":d}
                    the_file.write(json.dumps(to_write)+'\n')
            elif p < 1:
                with open(valid_json, 'a') as the_file:
                    to_write = {"fold":"valid","topic":"","dialogue":d}
                    the_file.write(json.dumps(to_write)+'\n')
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convertIEMOCAPjson.py")
    parser.add_argument("--input", type=str, default="/home/n/nguyenpk/IEMOCAP_full_release")
    parser.add_argument("--output", type=str, default="/home/n/nguyenpk/CS6208/GNN_ERC/preprocessing/data_fts")
    args = parser.parse_args()
    create_json_file(args.input, args.output)