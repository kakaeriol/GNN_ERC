import os
conf = {}
BASE_PATH = "/home/n/nguyenpk/CS6208/GNN_ERC"
DATA_PATH = os.path.join(BASE_PATH, "preprocessing/data_fts")

#-- DATA FOLDER
conf["base_IEMOCAP_path"] = os.path.join(DATA_PATH, "IEMOCAP_features")
conf["data_IEMOCAP_path"] = os.path.join(conf["base_IEMOCAP_path"], 'IEMOCAP_features.pkl')
#
conf["raw_MELD_path"] = os.path.join(BASE_PATH, "data/MELD.Raw")
conf["base_MELD_path"] = os.path.join(DATA_PATH, "MELD_features")
conf["data_MELD_path"] = os.path.join(conf["base_MELD_path"], 'MELD_features.pkl')

#
conf["base_DailyDialogue_path"] = os.path.join(DATA_PATH, "DailyDialogue_features")
conf["idata_DailyDialogue_path"] = os.path.join(conf["base_DailyDialogue_path"], "dailydailog")
conf["data_DailyDialogue_path"] = os.path.join(conf["base_DailyDialogue_path"], 'DailyDialogue_features.pkl')


conf["glove_path"] = os.path.join(BASE_PATH, "data/glove.840B.300d.txt")
#---- pkl file
