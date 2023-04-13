import sys
sys.path.append("..")
import os
import preprocessing_token as pp
import preprocessing_raw as pr
import pandas as pd


from sysconf import conf

def main():
    print('Preprocessing MELD raw')
    meld_tool = pr.MELD_preprocessing()
    meld_raw = meld_tool.raw_MELD_DS_segment(conf["raw_MELD_path"])
    pd.to_pickle(meld_raw, conf["pickle_raw_MELD"])
    #-------------------------------------------------
    print('Preprocessing IEMOCAP raw') ## from the pickle file
    iemo_itool = pr.IEMOCAP_preprocessing()
    iemo_raw = iemo_itool.create_IEMOCAP_from_pkl(conf["pickle_IEMOCAP_path"])
    pd.to_pickle(iemo_raw, conf["pickle_raw_IEMOCAP"])
    #-------------------------------------------------------------------------
    print('Preprocessing Dialog Daily raw') 
    dd_tool = pr.Daily_Dialog_preprocessing()
    diaglog_raw = dd_tool.raw_DD_DS_segment(conf["raw_Daily_Dailog_path"])
    pd.to_pickle(diaglog_raw, conf["pickle_raw_Daily_Dailog"])
    
    
    print('GENERAING THE TOKEN')
    for i in ["MELD", "IEMOCAP", "Daily_Dailog"]:
        print("Processing Token {} ".format(i))
        ikey_input = "pickle_raw_{}".format(i)
        ikey_out_token = "pickle_fts_token_{}".format(i)
        preprocess_tool = pp.Preprocessing_Data_Token(conf[ikey_input])
        _ = preprocess_tool.processing_token_data_segment(conf[ikey_out_token])
        print("Finish calculate token and save in {}".format(conf[ikey_out_token]))
        ikey_out_embedd = "pkl_embedd_{}".format(i)
        print("COMPUTING THE EMBEDDING MATRIX")
        _ =preprocess_tool.create_pretrain_embedding(conf[ikey_out_embedd])
        print("Finish embedding and save in: \n :{}".format(conf[ikey_out_embedd]))

        
if __name__ == "__main__":
    
    main()