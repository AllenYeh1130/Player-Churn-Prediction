#%%
# 建立lgbm模型，21天行為預測玩家是否會流失(7天沒上)，將需要的資訊傳到MLflow上
project_name = 'modeling'

### 載入套件
import datetime as dt
import os
import shutil
import socket
import sys
import time
import json
import numpy as np
import optuna
import pandas as pd
import sqlalchemy
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import mlflow


#%% config
# ===== config =====
### 取得ip位置，用來判斷要連的DB與Mlflow
# 先獲得主機位置，要判斷是正式機還是測試機
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    return ip_address

ip_address = get_ip_address()

### config
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

config = config_read(file_name = 'all_setting.json')


#%% DB connect
# ===== DB connect =====
# AI正式機資料庫連線
DATABASE = 'game'

# AI正式機資料庫連線
if ip_address == '正式機':
    # AI正式機
    USER = config['DB.formal.user']
    PASSWORD = config['DB.formal.pass']
    HOST = config['DB.formal.host']
    PORT = config['DB.formal.port']
elif ip_address == '測試機':
    # AI測試機內網
    USER = config['DB.test.user']
    PASSWORD = config['DB.test.pass']
    HOST = config['DB.test.host']
    PORT = config['DB.test.port']
else:
    # 本地端
    USER = config['DB.local.user']
    PASSWORD = config['DB.local.pass']
    HOST = config['DB.local.host']
    PORT = config['DB.local.port']
    
engine_stmt = 'mysql+pymysql://%s:%s@%s:%s/%s' % (USER,PASSWORD,HOST,PORT,DATABASE)
engine = sqlalchemy.create_engine(engine_stmt, echo=False)


#%% MLflow設定
# ===== MLflow設定 =====
experiment_name = 'experiment_name'
model_name = 'model_name'

# 設置tracking server uri、帳號密碼
# 以ip位置判斷MLflow該連正式還是測試
if ip_address == '正式機':         # AI正式機
    trackingServerUri = config['mlflow.mlflow_formal.trackingServerUri']
    os.environ["MLFLOW_TRACKING_USERNAME"] = config['mlflow.mlflow_formal.user_name']
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config['mlflow.mlflow_formal.password']
else:                                      # 非正式機就連到測試的MLflow
    trackingServerUri = config['mlflow.mlflow_test.trackingServerUri']
    os.environ["MLFLOW_TRACKING_USERNAME"] = config['mlflow.mlflow_test.user_name']
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config['mlflow.mlflow_test.password']
mlflow.set_tracking_uri(trackingServerUri)

# MLflow的experiment
mlflow.set_experiment(experiment_name)

# wait_until_ready函式
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name = model_name,
      version = model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)


#%% 設定日期
# ===== 設定日期 =====
arg = sys.argv
if len(arg) > 1:
    start_date = arg[1]
    end_date = arg[2]
else:
    # 沒設定日期，預設抓90天的資料來建模
    start_date = (dt.date.today() - dt.timedelta(days=96)).strftime('%Y-%m-%d')
    end_date = (dt.date.today() - dt.timedelta(days=7)).strftime('%Y-%m-%d')

# start_date不能晚於end_date
if (start_date > end_date):
    start_date = end_date

print(f'建模資料日期： {start_date} ~ {end_date}')


#%% 初次lgbm建模，評估特徵重要性
# ===== 初次建模，初步判斷特徵重要性 =====
### UnderSampling & 資料拆訓練集、測試集
# per是留存用戶會乘的比例，不是直接代表流失占比 = per
# 建立訓練資料
sql = """SELECT * FROM game_info WHERE date BETWEEN '%s' AND '%s';""" %(start_date, end_date)        
all_data = pd.read_sql(sql, engine)
print('資料抓取成功')

# 區分train, test
X_train, X_test, y_train, y_test = train_test_split(all_data.drop(['lost', 'account_id', 'date'], axis = 1), 
                                                    all_data['lost'], 
                                                    stratify = all_data['lost'], shuffle = True, random_state = 0)

# UnderSampling
# train留存用戶只留50%，避免留存用戶過多造成資料不平衡
per = 0.5
X_train['lost'] = y_train
x_a = X_train[X_train['lost'] == 1]     #流失用戶
x_b = X_train[X_train['lost'] == 0]     #留存用戶
X_train = pd.concat([x_a, x_b.sample(round(len(x_b) * per))], axis=0).sample(frac = 1)
y_train = X_train['lost']
X_train.drop('lost', axis=1, inplace=True)


### 用Optuna尋找最佳化超參數
def objective(trial, X, y):
    
    param = {
        # 'boosting': trial.suggest_categorical("boosting", ["dart", "gbdt"]),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'n_estimators': trial.suggest_int('n_estimators', 600, 900),
        'num_leaves': trial.suggest_int('num_leaves', 100, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 25),
        #'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        #'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.8, 3),
        'subsample': trial.suggest_loguniform('subsample', 0.7, 1),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.6, 0.9)
    }
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    
    metric_test = []
    
    for train_index, test_index in cv.split(X, y):
        # 拆訓練組、測試組
        X_train_opt, X_test_opt = X.iloc[train_index], X.iloc[test_index]
        y_train_opt, y_test_opt = y.iloc[train_index], y.iloc[test_index]
        
        # weight
        # LGBM中類似underSampling的功能，用來處理不平衡資料
        # 如果要測試weight就把per設成1，不然underSampling時會流失部分資料
        # weight = y_train_opt.value_counts()[0]/y_train_opt.value_counts()[1]
        # Fit the model
        # optuna_model = LGBMClassifier(scale_pos_weight = weight, **param)
        
        # Fit the model
        optuna_model = LGBMClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt, eval_metric='logloss')
        # optuna_model.fit(X_train_opt, y_train_opt, early_stopping_rounds=30, 
        #                   eval_set=[(X_train_opt, y_train_opt), (X_test_opt, y_test_opt)])
    
        # 使用model預測的結果
        pred_test = optuna_model.predict(X_test_opt)
        
        # 優化目標:設為log_loss
        metric = log_loss(y_test_opt, pred_test)
        
        metric_test.append(metric)

    return np.mean(metric_test)
    
# Creating Optuna object and defining its parameters
study = optuna.create_study(direction='minimize')
func = lambda trial: objective(trial, X = X_train, y = y_train)
study.optimize(func, n_trials = 10, n_jobs = -1)
# study.optimize(func, n_trials = 10, n_jobs = 2)

### Feature Selection
# 使用最佳參數建模
lgbm = LGBMClassifier(**study.best_params)
lgbm.fit(X_train, y_train)

# 獲取特徵重要性
importance_scores = lgbm.feature_importances_

# 將特徵名稱和重要性分數組合起來
feature_importance = list(zip(X_train.columns, importance_scores))
# 篩選前40個重要的特徵
feature_importance = pd.DataFrame(sorted(feature_importance, key=lambda x: x[1], reverse=True))
feature_importance = feature_importance.loc[0:39, 0].append(pd.Series(['lost', 'account_id', 'date']))
all_data = all_data[feature_importance]
     

#%% lgbm建模找超參數   
# ===== 再次建模，找lgbm超參數 =====
### UnderSampling & 資料拆訓練集、測試集
per = 0.5
X_train, X_test, y_train, y_test = train_test_split(all_data.drop(['lost', 'account_id', 'date'], axis = 1), 
                                                    all_data['lost'], 
                                                    stratify = all_data['lost'], shuffle = True, random_state = 0)
X_train['lost'] = y_train
x_a = X_train[X_train['lost'] == 1]
x_b = X_train[X_train['lost'] == 0]

X_train = pd.concat([x_a, x_b.sample(round(len(x_b) * per))], axis=0).sample(frac = 1)
y_train = X_train['lost']
X_train.drop('lost', axis=1, inplace=True)

### 用Optuna尋找最佳化超參數
def objective_2(trial, X, y):
    
    param = {
        # 'boosting': trial.suggest_categorical("boosting", ["dart", "gbdt"]),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'n_estimators': trial.suggest_int('n_estimators', 600, 900),
        'num_leaves': trial.suggest_int('num_leaves', 100, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 25),
        #'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        #'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.8, 3),
        'subsample': trial.suggest_loguniform('subsample', 0.7, 1),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.6, 0.9)
    }
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    
    metric_test = []
    
    for train_index, test_index in cv.split(X, y):
        # 拆訓練組、測試組
        X_train_opt, X_test_opt = X.iloc[train_index], X.iloc[test_index]
        y_train_opt, y_test_opt = y.iloc[train_index], y.iloc[test_index]
        
        # weight
        # weight = y_train_opt.value_counts()[0]/y_train_opt.value_counts()[1]
        
        # Fit the model
        # optuna_model = LGBMClassifier(scale_pos_weight = weight, **param)
        optuna_model = LGBMClassifier(**param)
        optuna_model.fit(X_train_opt, y_train_opt, eval_metric='logloss')
        # optuna_model.fit(X_train_opt, y_train_opt, early_stopping_rounds=30, 
        #                   eval_set=[(X_train_opt, y_train_opt), (X_test_opt, y_test_opt)])
    
        # 使用model預測的結果
        pred_test = optuna_model.predict(X_test_opt)
        
        # 優化目標:設為log_loss
        metric = log_loss(y_test_opt, pred_test)
        metric_test.append(metric)

    return np.mean(metric_test)

# Creating Optuna object and defining its parameters
study = optuna.create_study(direction='minimize')
func = lambda trial: objective_2(trial, X = X_train, y = y_train)
study.optimize(func, n_trials = 100, n_jobs = -1)
# study.optimize(func, n_trials = 50, n_jobs = 2)


#%% 紀錄模型訓練結果
# ===== 紀錄模型訓練結果 =====
# 創建temp資料夾
path = os.path.join(os.getcwd(), "temp")
if not os.path.exists(path):
    os.mkdir(path)

with mlflow.start_run():
    ### 使用最佳參數建模
    lgbm = LGBMClassifier(**study.best_params)
    lgbm.fit(X_train, y_train)
    
    # 匯出模型model.pkl
    import joblib
    joblib.dump(lgbm, 'model.pkl')
    
    ### 計算模型指標
    f1 = round(f1_score(y_test, lgbm.predict(X_test)),3)
    recall = round(recall_score(y_test, lgbm.predict(X_test)),3)
    precision = round(precision_score(y_test, lgbm.predict(X_test)),3)
    accuracy = round(accuracy_score(y_test, lgbm.predict(X_test)),3)
    # 匯出訓練結果 (txt)
    with open('model_result.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write(f"F1 Score    : {f1:.3f}\n")
        f.write(f"Recall      : {recall:.3f}\n")
        f.write(f"Precision   : {precision:.3f}\n")
        f.write(f"Accuracy    : {accuracy:.3f}\n")

    # 匯出特徵資訊 (CSV)
    importance_scores = pd.DataFrame({'features' : lgbm.feature_name_, 'importances' : lgbm.feature_importances_})
    importance_scores.to_csv('features.csv', index=False)

    ### MLflow寫入模型、metrics
    mlflow.sklearn.log_model(lgbm, "model", registered_model_name= model_name, serialization_format='pickle')
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1",f1)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
        
    # temp資料夾中創temp_package資料夾
    os.mkdir(os.path.join(path,"temp_package"))

    # 把檔案存進temp_package資料夾中
    for file in ["data_processing.py", "modeling.py", "predict.py", "features.csv"]:
        if os.path.exists(os.path.join(os.getcwd(), file)):
            shutil.copy(os.path.join(os.getcwd(), file), os.path.join(path, "temp_package", file))
                 
    # 把temp資料的檔案(包括temp_package資料夾)丟到 ./mount/mlflow_store/...
    mlflow.log_artifacts(path)

# 抓取MLflow上的版本資訊
client = MlflowClient()
model_version_infos = client.search_model_versions("name = '%s'" % model_name)

# 找出最新版本，跑wait_until_ready
new_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
wait_until_ready(model_name, new_model_version)

# Update最新版本
client.update_model_version(
name=model_name,
version=new_model_version,
description="This model version is new version."
)

# 把最新版本的模型的階段改成Production
client.transition_model_version_stage(
name=model_name,
version=new_model_version,
stage="Production"
)

mlflow.end_run()
if os.path.exists(path):
    shutil.rmtree(path) 
# with open('model.pkl', 'wb') as files:
#     pkl.dump(automl, files)



