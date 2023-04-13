#  Create dataset from MELD
from typing import List, Union, Any
import os
import numpy as np
import re
import pandas as pd
class Daily_Dialog_preprocessing:
    """This class 
    just use to wrap up
    the function"""
    def my_reading_conv(self, dpath: str, 
                        lsplit: str=" ", to_int: bool=False) -> List[Any]:
        """ This function reading from dpath and return the output
        Input:
         - dpath: the path of data
         - split: split parameters for every line 
        Output: list of the input from dpath. 
        """

        with open(dpath, "r") as f:
            if to_int:
                lines = [[int(i.strip()) for i in line.rstrip().split(lsplit) if i.strip() != ''] for line in f ]
            else:
                lines = [[i.strip() for i in line.rstrip().split(lsplit) if i.strip() != ''] for line in f ]
        return lines

    def reading_DD(self, conv_path: str, 
                 emo_path: str, 
                 act_path: Union[str, None]=None, 
                 topic_path: Union[str, None]=None) -> dict:
        """ This function is create the dataset from multiple path of Daily Dailog
        Input:
            - conv_path: path of conversation data
            - emo_path: path of the emotional data
            - act_path: path of action data
            - topic_path: path of topic data
        Output: The dictionary the have 
            {
            "conv": the list of the converstion[each conversation have number of ulterance]
            "speakers": output the auto generate speaker based on the id of conv
            "emo" : the list of emotion
            "act":  the list of act
            "topic": the list of topic
            }
        """
        conversations = self.my_reading_conv(conv_path, "__eou__")
        emotions = self.my_reading_conv(emo_path, to_int=True)
        speakers = [list(map(lambda i: i%2, range(0, len(ic)))) for ic in conversations]
        ### This process to check if the length when we create is different ###
        len_conv = list(map(lambda x: len(x), conversations))
        len_emo = list(map(lambda x: len(x), emotions))
        check1 = all(item in len_conv for item in len_emo)
        assert check1, "Mismatch label and data"
        len_speaker = list(map(lambda x: len(x), speakers))
        check2= all(item in len_speaker for item in len_emo), "Mismatch speaker and data"
        acts = []
        topics =[]
        if act_path:
            acts = self.my_reading_conv(act_path, to_int=True)
            len_acts = list(map(lambda x: len(x), acts))
            check2= all(item in len_acts for item in len_emo)
            assert check2, "Mismatch actions and data"
        if topic_path:
            topics =  self.my_reading_conv(topic_path, to_int=True)
            topics = list(map(lambda x: x[0], acts))
            assert check2, "Mismatch topic and data"
        out = {'conversation': conversations, 'speakers':speakers,  'emotions': emotions, 'actions':acts, 'topics':topics}
        return out

    def raw_DD_DS_segment(self, basepath: str)->dict:
        """ This function will recevide the base path and read 
        for train, test, valid for each data 
        Input:
            - basepath: the path the have subpath 'train', 'valid' and 'test' inside
        Output: dictionary of every 'train', 'test' and 'valid'
            - rs {
            'train': {'conversations', 'speakers', 'emotions', 'actions', 'topics'}
            'test': {'conversations', 'speakers', 'emotions', 'actions', 'topics'}
            'valid': {'conversations', 'speakers', 'emotions', 'actions', 'topics'}
            }
        """
        def create_subpath(substr: str) -> List[str]: 
            """ return the sub path base on substr """
            conv_path = os.path.join(basepath, substr, 'dialogues_{}.txt'.format(substr))
            emo_path  = os.path.join(basepath, substr, 'dialogues_emotion_{}.txt'.format(substr))
            act_path  = os.path.join(basepath, substr, 'dialogues_act_{}.txt'.format(substr))
            return conv_path, emo_path, act_path
        train_conv, train_emo, train_act = create_subpath('train')
        test_conv, test_emo, test_act = create_subpath('test')
        valid_conv, valid_emo, valid_act = create_subpath('validation')
        out_rs = {'train': self.reading_DD(train_conv, train_emo, train_act),
                  'test': self.reading_DD(test_conv, test_emo, test_act),
                  'dev': self.reading_DD(valid_conv, valid_emo, valid_act),
                 }
        return out_rs

class MELD_preprocessing:
    """ 
    This class just keep all of the pre processing MELD script in one sample
    """
    def __init__(self):
        self.mapping_emo_label_MELD = {
                'neutral': 0, 
                'anger':1, 
                'disgust':2, 
                'fear': 3,
                'joy': 4, 
                'sadness':5, 
                'surprise':6,
            }
    def read_from_MELD(self, path_link: str)->dict:
        """
        Input: the link to csv MELD dataset 
        Output: the preprocessing MELD
        """
        data = pd.read_csv(path_link)
        def preprocessingMELD_col(cols: str):
            """ 
            """
            uid = list(cols.Utterance_ID)
            ultereance = list(cols.Utterance)
            speaker = list(cols.Speaker)
            emotion = list(cols.Emotion)
            sr_out = {}
            sr_out['Utterance_ID'] = uid
            sr_out['conversation'] = ultereance
            sr_out['emotions_raw'] = emotion
            sr_out["emotions"] = [self.mapping_emo_label_MELD.get(i) for i in emotion]
            sr_out["Speaker_raw"] = speaker
            map_sp = {}
            isp = 0
            for ii in speaker:
                if ii not in map_sp.keys():
                    map_sp[ii] = isp
                    isp = isp + 1
            sr_out["speakers"] = [map_sp.get(i) for i in speaker]
            sr_out["num_speakers"] = max(map_sp.values()) + 1
            return pd.Series(sr_out)


        icols = [ 'Utterance_ID', 'Utterance', 'Speaker', 'Emotion']
        data_process = data.groupby('Dialogue_ID')[icols].apply(lambda x: preprocessingMELD_col(x)).reset_index()

        rs = {}
        for i in data_process.columns:
            rs[i] = data_process[i].to_list()
        return rs

    def raw_MELD_DS_segment(self, basepath: str)->dict:
        """ 
        This function output the train, test, valid of the MELD DATA set 
        Input: 
        - basepath: the string path, in this path will have these file 
            + dev_sent_emo.csv
            + train_sent_emo.csv
            + test_sent_emo.csv
        Output:
        - dictionary of 'train', 'test' and 'dev' test
        """
        train_path = os.path.join(basepath, "train_sent_emo.csv")
        test_path = os.path.join(basepath, "test_sent_emo.csv")
        valid_path = os.path.join(basepath, "dev_sent_emo.csv")
        rs = {}
        rs['train'] = self.read_from_MELD(train_path)
        rs['test']  = self.read_from_MELD(test_path)
        rs['dev'] = self.read_from_MELD(valid_path)
        rs['mapping_emo'] = self.mapping_emo_label_MELD
        return rs
    

class IEMOCAP_preprocessing: 
    """ 
    This class just keep all of the pre processing IEMOCAP  script in one 
    """
    def __init__(self, map_label={}):
        self.map_emo_label = {
            'hap':0, 
            'sad':1, 
            'neu':2, 
            'ang':3, 
            'exc':4, 
            'fru':5}
        self.map_speaker = {
            'M': 0, 
            'F': 1
        }
        self.map_emo_label.update(map_label)
        
    def create_IEMOCAP_from_pkl(self, pkl_file: str) ->dict:
        """
        This function output the train, test, valid of the IEMOCAP DATA set 
            Input: 
            - pkl_file: IEMOCAP_feature.pickle file the author used
            Output:
            - dictionary of 'train', 'test' and 'valid' test
        """
        videoIDs, videoSpeakers, videoLabels, \
        _, _, _, videoSentence, trainVid, testVid  = pd.read_pickle(pkl_file)
        dev_size = int(len(trainVid)*0.1) ### as their method
        train_video, valid_video = trainVid[dev_size:], trainVid[:dev_size]
        def get_data(list_idx: List[str])->dict:
            conv = []
            speaks = []
            emos = []
            #----
            for idx in list_idx:
                conv.append(videoSentence[idx])
                speaks.append([self.map_speaker.get(i) for i in videoSpeakers[idx]])
                emos.append([self.map_emo_label.get(i, i) for i in videoLabels[idx]]) #in case it already transform use itself
            rs = {'conversation': conv, 'speakers': speaks, 'emotions':emos}
            return rs
        train = get_data(train_video)
        test = get_data(testVid)
        valid = get_data(valid_video)
        out_rs = {'train': train, 'dev': valid, 'test': test, 'train_idx': train_video,
                  'dev_idx':valid_video, 'test_idx':test, 'mapping_emo': self.map_emo_label}
        
        return out_rs
    
    def create_dataset_from_IEMOCAP_base(self, base_path:str)->dict:
        """
        In the original dataset, there is some 10 type of emotions, 
        including: 
        Neu = neutral state, Hap = happiness, Sad = sadness, Ang = anger,
        Sur = surprise, Fea = fear, Dis = disgust, Fru = frustation, Exc = excited and
        Oth = other
        {'ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'oth', 'sad', 'sur', 'xxx'}
        I found that 'xxx' mean 'neu' (coz there is several sentense in this case, 
        and its sensentce having it own emotions)
        base_path of the IEMOCAP
        """
        emo_path = os.path.join(base_path, '{}', 'dialog/EmoEvaluation')
        trans_path = os.path.join(base_path, '{}', 'dialog/transcriptions')
        sess = [f for f in os.listdir(base_path) if re.match(r'Ses', f)]
        dailog_id = []
        conversation = []
        speakers = []
        emotions = []
        def find(x, lines):
            """Find the line have content x"""
            for ii in lines:
                if x in ii:
                    return ii
            return False

        def finding_sub_pattern(ifile_trans:str, ifile_emo:str)->List[Any]:
            """
            This return the one conversation information 
            """

            uid = []
            speaker =  []
            conv = []
            emo = []
            count = 0
            with open(ifile_emo, "r") as f:
                emo_lines = f.readlines()

            with open(ifile_trans, "r") as f:
                lines = f.readlines()
            for line in lines:
                iuid, iuterance = line.split(":")
                if "Ses" not in iuid: #remvoe some action and unidentify sensentce label
                    continue
                iuid = iuid.split(" ")[0]
                eline = find(iuid, emo_lines)
                if not eline:
                    continue 
                temp = iuid.split('_')
                uid.append(iuid)
                speaker.append(self.map_speaker.get(temp[-1][0]))
                conv.append(iuterance.strip())
                emo.append(eline.split('\t')[2])
            return uid, conv, speaker, emo

        #
        for isess in sess:
            link_Emo = emo_path.format(isess)
            link_transcript = trans_path.format(isess)
            files = [f for f in os.listdir(link_transcript) if re.match(r'Ses+.*\.txt', f)]
            for ifile in files:
                ifile_trans = os.path.join(link_transcript, ifile)
                ifile_emo = os.path.join(link_Emo, ifile)
                uid, conv, speak, emo = finding_sub_pattern(ifile_trans, ifile_emo)
                dailog_id.append(uid)
                conversation.append(conv)
                speakers.append(speak)
                emotions.append(emo)
                assert len(uid) == len(conv), "Check the input"
                assert len(uid) == len(speak) , "Check the input"
                assert len(uid) == len(emo), "Check the input {} {}".format(len(uid), len(emo))

        out = {'dailog_id': dailog_id, 'conversation': conversation, 'speakers':speakers, 'emotions':emotions}
        return out