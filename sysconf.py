import os
conf = {}
BASE_PATH = "/home/n/nguyenpk/CS6208/GNN_ERC"
DATA_PATH = os.path.join(BASE_PATH, "preprocessing/data_fts")

# #-- DATA FOLDER
# conf["base_IEMOCAP_path"] = os.path.join(DATA_PATH, "IEMOCAP_features")
# conf["data_IEMOCAP_path"] = os.path.join(conf["base_IEMOCAP_path"], 'IEMOCAP_features.pkl')
# #
# conf["raw_MELD_path"] = os.path.join(BASE_PATH, "data/MELD.Raw")
# conf["base_MELD_path"] = os.path.join(DATA_PATH, "MELD_features")
# conf["data_MELD_path"] = os.path.join(conf["base_MELD_path"], 'MELD_features.pkl')

# #
# conf["base_DailyDialogue_path"] = os.path.join(DATA_PATH, "DailyDialogue_features")
# conf["idata_DailyDialogue_path"] = os.path.join(conf["base_DailyDialogue_path"], "dailydailog")
# conf["data_DailyDialogue_path"] = os.path.join(conf["base_DailyDialogue_path"], 'DailyDialogue_features.pkl')


# conf["glove_path"] = os.path.join(BASE_PATH, "data/glove.840B.300d.txt")
# #---- pkl file

conf["raw_MELD_path"] = "/home/n/nguyenpk/CS6208/GNN_ERC/data/MELD.Raw"
conf["raw_Daily_Dailog_path"] = "/home/n/nguyenpk/CS6208/GNN_ERC/data/Dialog" 
conf["raw_IEMOCAP_path"] = "/home/n/nguyenpk/IEMOCAP_full_release"
conf["pickle_IEMOCAP_path"] = "/home/n/nguyenpk/IEMOCAP_features.pkl"
#
conf["pickle_raw_MELD"] = "/home/n/nguyenpk/CS6220/project/NLP_DS_distance/data/MELD_raw.pkl"
conf["pickle_raw_IEMOCAP"] = "/home/n/nguyenpk/CS6220/project/NLP_DS_distance/data/IEMOCAP_raw.pkl"
conf["pickle_raw_Daily_Dailog"] = "/home/n/nguyenpk/CS6220/project/NLP_DS_distance/data/Daily_Dailog_raw.pkl"
#
conf["pickle_fts_token_MELD"] = os.path.join(BASE_PATH, "NP_preprocessing/data/MELD_token_fts.pkl")
conf["pickle_fts_token_IEMOCAP"] = os.path.join(BASE_PATH, "NP_preprocessing/data/IEMOCAP_token_fts.pkl")
conf["pickle_fts_token_Daily_Dailog"] =  os.path.join(BASE_PATH, "NP_preprocessing/data/Daily_token_fts.pkl")
# ----------
conf["pkl_embedd_MELD"] =  os.path.join(BASE_PATH, "NP_preprocessing/data/MELD_embedd.pkl")
conf["pkl_embedd_IEMOCAP"] =  os.path.join(BASE_PATH, "NP_preprocessing/data/IEMOCAP_embedd.pkl")
conf["pkl_embedd_Daily_Dailog"] =  os.path.join(BASE_PATH, "NP_preprocessing/data/Daily_embedd.pkl")
#
conf["glove_path"] = "/home/n/nguyenpk/CS6220/data/glove.840B.300d.txt"