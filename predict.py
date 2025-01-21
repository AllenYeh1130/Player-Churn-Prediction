#%%
# 撈取預測用資料，與模型(pkl)進行預測並處理預測出的結果
project_name = 'predict'

import pandas as pd
import datetime as dt
import sys
import warnings
import os
import json
from data_processing import data_processing
import pickle as pkl
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
parser_n = 10000
game = 'game'
# model_path = 'model.pkl的路徑'


def predict_fun(model_path):
    #%% 設定
    # ===== config =====
    # 資料寫入、迴圈處理每個batch、每個batch處理1000筆資料
    def config_read(file_name):
        # Read config file
        if sys.platform.startswith('linux'):
            file_path = r'linux 設定檔路徑'  # linux文件路径
        elif sys.platform.startswith('win'):
            file_path = r'Windows 設定檔路徑'
        else:
            print("無法判斷程式執行的作業系統")

        file_path = os.path.join(file_path, file_name) #完整設定檔路徑
        #讀入json
        with open(file_path, 'r') as file:
            config = json.load(file)
        
        config = pd.json_normalize(config).stack().reset_index(level=0, drop=True) #刪除多的一層索引
        config = config.to_dict()
        return config

    API_config = config_read(file_name = 'API_setting.json')
    token = API_config[f'{game}.token']
    

    #%% 撈預測資料
    # token & project_id
    token = API_config[f'{game}.token']
    project_id = API_config[f'{game}.project_id']
    
    # 設定日期
    today = str(dt.date.today())
    
    # 呼叫data_processing
    # 美服時區為UTC-5
    data = data_processing(token = token, start_date = today, end_date = today, time_diff = 0, project_id = project_id)
    # account_id轉為字串
    data['account_id'] = data['account_id'].astype(str)
    
    
    #%% 預測資料並區分流失及流失風險(高中低)
    # 流失機率分高中低
    def map_proba(x):
        if x >= 0.8:
            return "high"
        elif x < 0.8 and x >= 0.65:
            return "medium"
        else:
            return "low"
    
    # 讀取model
    with open(model_path, 'rb') as f:
            model = pkl.load(f)
    
    # 留下帳號
    data['account_id'] = data['account_id'].astype(str)
    acc_id = data['account_id']
    # 選取特徵
    feature = model.feature_name_
    data = data[feature]
    # 預測結果
    pred = model.predict(data)
    # 流失機率
    pred_proba = model.predict_proba(data)[:,1]
    
    # 整合資料
    result = pd.DataFrame({'account_id' : acc_id, 'pred' : pred, 'pred_proba' : pred_proba})
    result = result[result['pred'] == 1].drop(columns=['pred'])
    result['pred_class'] = result['pred_proba'].apply(lambda x:map_proba(x))
    
    # 分A、B組，根據流失機率高中低類別均勻分配
    try:
        A, B = train_test_split(result, test_size=0.5, random_state=0, stratify = result["pred_class"])
        A['group'] = 'A'
        B['group'] = 'B'
        result = pd.concat([A,B], axis=0)
    except:
        result['group'] = 'A'
        
    # 匯出預測結果(CSV)
    result.to_csv('predict_result.csv', index=False)
        
    return result