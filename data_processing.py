#%%
# 從url抓資料 & 資料處理主程式
import pandas as pd
import datetime as dt
import numpy as np
import warnings
import urllib
import requests
from io import StringIO
warnings.filterwarnings("ignore")

# =============================================================================
# time_diff : 時差(int)
# project_id : 遊戲id，在API_config中有設定
# =============================================================================

def data_processing(token, start_date, end_date, time_diff, project_id):
    #%%
    # 設定時間，預測的前21天內行為
    start_str = ' 0'+str(time_diff)+':00:00' if time_diff <= 10 else ' '+str(time_diff)+':00:00'
    end_str = ' 0'+str(time_diff-1)+':59:59' if time_diff <= 10 else ' '+str(time_diff-1)+':59:59'
    end_str = end_str if time_diff != 0 else ' 23:59:59'
    start_time = (pd.to_datetime(start_date) - dt.timedelta(days=21)).strftime('%Y-%m-%d') + start_str
    end_time = (pd.to_datetime(end_date)).strftime('%Y-%m-%d') + end_str
    end_time_for_lost = (pd.to_datetime(end_date) + dt.timedelta(days=7)).strftime('%Y-%m-%d') + start_str  # 判斷是否流失的時間 

    #%%
    # ==== 每日在線時長 ====
    # online_time原單位為秒，處理後轉為分
    # 單次上線超過3小時判斷為異常，調整為3小時
    SQL = '''/* $part_date */
    SELECT account_id, real_date, sum(online_time) as online_time FROM (
        SELECT "#account_id" as account_id, 
                DATE(date_add('hour', -%s, "#event_time")) as real_date, 
                CASE WHEN online_time/60 >=180 THEN 180
                     ELSE online_time/60 
                END as online_time
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('logout'))
        AND (online_time >= 0) )
    GROUP BY account_id, real_date
    ''' %(time_diff, project_id, start_time, end_time_for_lost)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s=str(r.content,'utf-8')
    online_data = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日首登時間(5點後) ====
    SQL = '''/* $part_date */
    SELECT account_id, real_date, min(login_time) as first_hour FROM (
        SELECT "#account_id" as account_id, 
        DATE(date_add('hour', -%s, "#event_time")) as real_date, 
        HOUR(date_add('hour', -%s, "#event_time")) as login_time
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('login'))
        AND (HOUR(date_add('hour', -%s, "#event_time")) >= 5) )
    GROUP BY account_id, real_date
    ''' %(time_diff, time_diff, project_id, start_time, end_time, time_diff)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s=str(r.content,'utf-8')
    first_hour = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日是否登入 ====
    SQL = '''/* $part_date */
    SELECT DISTINCT "#account_id" as account_id, 
            DATE(date_add('hour', -%s, "#event_time")) as real_date, 
            1 as login_counts
    FROM game_%s_data
    WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
    AND ("$part_event" IN ('login'))
    ''' %(time_diff, project_id, start_time, end_time)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s=str(r.content,'utf-8')
    login_counts = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日排位賽/非排位賽的遊玩時長 ====
    SQL = '''/* $part_date */
    SELECT starts.account_id, starts.real_date, start_time, starts.type_game, min(date_diff('second', CAST(start_time as timestamp), CAST(end_time as timestamp))) as game_time
    FROM(
        SELECT "#account_id" as account_id, "#event_time" as start_time, ts as start_ts, DATE(date_add('hour', -%s, "#event_time")) as real_date, REPLACE("#event_name", '_start', '') as type_game
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('game_unranked_start', 'game_ranked_start'))
    ) starts
    LEFT JOIN (
        SELECT "#account_id" as account_id, "#event_time" as end_time, ts as end_ts, DATE(date_add('hour', -%s, "#event_time")) as real_date, REPLACE("#event_name", '_end', '') as type_game
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('game_unranked_end', 'game_ranked_end'))
    ) ends
    ON starts.account_id = ends.account_id AND starts.real_date = ends.real_date AND starts.type_game = ends.type_game
    WHERE (start_time < end_time)
    GROUP BY starts.account_id, starts.real_date, starts.type_game, start_time
    ''' %(time_diff, project_id, start_time, end_time, time_diff, project_id, start_time, end_time)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s = str(r.content,'utf-8')
    game = pd.read_csv(StringIO(s))
    
    # 修改遊戲型態的字尾
    game['type_game'] = game['type_game'] + '_time'
    
    # 時長加總、分排位賽.非排位賽
    game_time = game.groupby(by=['account_id', 'real_date', 'type_game']).sum().reset_index()
    game_time = game_time.pivot(index = ['account_id', 'real_date'], columns = 'type_game', values = 'game_time').reset_index().rename_axis(None, axis=1)
    
    #%%
    # ==== 每日排位賽/非排位賽的間隔時長 ====
    SQL = '''/* $part_date */
    SELECT starts.account_id, starts.real_date, start_time, starts.type_game, min(date_diff('second', CAST(start_time as timestamp), CAST(end_time as timestamp))) as game_next_time
    FROM(
        SELECT "#account_id" as account_id, "#event_time" as start_time, ts as start_ts, DATE(date_add('hour', -%s, "#event_time")) as real_date, "#event_name" as type_game
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('game_unranked_start', 'game_ranked_start'))
    ) starts
    LEFT JOIN (
        SELECT "#account_id" as account_id, "#event_time" as end_time, ts as end_ts, DATE(date_add('hour', -%s, "#event_time")) as real_date, "#event_name" as type_game
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('game_unranked_start', 'game_ranked_start'))
    ) ends
    ON starts.account_id = ends.account_id AND starts.real_date = ends.real_date AND starts.type_game = ends.type_game
    WHERE (start_time < end_time)
    GROUP BY starts.account_id, starts.real_date, starts.type_game, start_time
    ''' %(time_diff, project_id, start_time, end_time, time_diff, project_id, start_time, end_time)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s = str(r.content,'utf-8')
    game = pd.read_csv(StringIO(s))
    
    # 修改遊戲型態的字尾
    game['type_game'] = game['type_game'].str.replace('_start', '_next_time')
    # game['type_game'] = game['type_game'].str.replace('_end', '_time')
    
    # 時長加總、分排位賽.非排位賽
    game_next_time = game.groupby(by=['account_id', 'real_date', 'type_game']).sum().reset_index()
    game_next_time = game_next_time.pivot(index = ['account_id', 'real_date'], columns = 'type_game', values = 'game_next_time').reset_index().rename_axis(None, axis=1)
    
    #%%
    # ==== 每日排位賽/非排位賽的次數 & 勝場次數 ====
    SQL = '''/* $part_date */
    SELECT "#account_id" as account_id, game_result, DATE(date_add('hour', -%s, "#event_time")) as real_date, "#event_name" as type_game
    FROM game_%s_data
    WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
    AND ("$part_event" IN ('game_unranked_end', 'game_ranked_end')) 
    ''' %(time_diff, project_id, start_time, end_time)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header')
    s = str(r.content,'utf-8')
    game_result = pd.read_csv(StringIO(s))
    
    # 修改遊戲型態的字尾
    game_result['type_game'] = game_result['type_game'].str.replace('_end', '_counts')
    
    # 計算次數、分排位賽.非排位賽
    game_counts = game_result.groupby(['account_id', 'real_date', 'type_game']).size().reset_index(name='counts')
    game_counts = game_counts.pivot(index = ['account_id', 'real_date'], columns = 'type_game', values = 'counts').reset_index().rename_axis(None, axis=1)
    
    # # 計算勝場次數、分排位賽.非排位賽
    # game_win_counts = game_result[game_result['game_result']=='勝利'].groupby(['account_id', 'real_date', 'type_game']).size().reset_index(name='win_counts')
    # game_win_counts = game_win_counts.pivot(index = ['account_id', 'real_date'], columns = 'type_game', values = 'win_counts').reset_index().rename_axis(None, axis=1)
    # game_win_counts.rename(columns = {'game_ranked_counts' : 'game_ranked_win_counts', 
    #                                   'game_unranked_counts' : 'game_unranked_win_counts'}, inplace = True)
    
    #%%
    # ==== 每日最後一次登出時的level & level_training ====
    SQL = '''/* $part_date */
    SELECT account_id, real_date, max(level) as level, max(level_training) as level_training FROM (
        SELECT "#account_id" as account_id, 
                DATE(date_add('hour', -%s, "#event_time")) as real_date, 
                level, 
                level_training
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp))
        AND ("$part_event" IN ('logout')) )
    GROUP BY account_id, real_date
    ''' %(time_diff, project_id, start_time, end_time)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded'},
                      data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    level = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日是否有development_talent ==== 
    SQL = '''/* $part_date */
    SELECT DISTINCT "#account_id" as account_id, 
    DATE(date_add('hour', -%s, "#event_time")) as real_date, 
    1 as development_talent
    FROM game_%s_data
    WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
    AND ("$part_event" IN ('development_talent'))
    ''' %(time_diff, project_id, start_time, end_time)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    development_talent = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日任務完成次數 ====
    SQL = '''/* $part_date */
    SELECT account_id, real_date, count() as quest FROM (
        SELECT "#account_id" as account_id, 
        DATE(date_add('hour', -%s, "#event_time")) as real_date
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp))
        AND ("$part_event" IN ('daily_quest'))
    )
    GROUP BY account_id, real_date
    ''' %(time_diff, project_id, start_time, end_time)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    quest = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 每日廣告觀看次數 ==== 
    SQL = '''/* $part_date */
    SELECT account_id, real_date, max(iaa_daily_count) as iaa_daily_count FROM (
        SELECT "#account_id" as account_id, 
                DATE(date_add('hour', -%s, "#event_time")) as real_date, 
                iaa_daily_count
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)) 
        AND ("$part_event" IN ('iaa_end'))
        AND (iaa_result='成功')
        AND (iaa_type='reward')
        )
    GROUP BY account_id, real_date
    ''' %(time_diff, project_id, start_time, end_time)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    iaa_count = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 鑽石消耗數 ====
    SQL = '''/* $part_date */
    SELECT account_id, real_date, sum(diamond_consume_amount) as diamond_consume FROM (
        SELECT "#account_id" as account_id, 
        DATE(date_add('hour', -%s, "#event_time")) as real_date, 
        diamond_consume_amount
        FROM game_%s_data
        WHERE ("#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp))
        AND ("$part_event" IN ('diamond_consume'))
        AND (diamond_consume_amount > 0) )
    GROUP BY account_id, real_date
    ''' %(time_diff, project_id, start_time, end_time)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    diamond = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 創角日期 ==== 
    # 創角幾日在每日迴圈中再計算
    SQL = '''
    SELECT "#account_id" as account_id, 
    DATE(date_add('hour', -%s, 
                  CASE WHEN create_role_time is null THEN register_time 
                  ELSE create_role_time END)) as create_date
    FROM user_%s_info
    WHERE regexp_like("#account_id", '^[0-9\.]+$')
    ''' %(time_diff, project_id)
    
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    create_data = pd.read_csv(StringIO(s))
    
    #%%
    # ==== 玩家的initial_channel ====
    # 在get_data使用
    SQL = '''
    SELECT "#account_id" as account_id, initial_channel
    FROM user_%s_info
    WHERE regexp_like("#account_id", '^[0-9\.]+$')
    ''' %(project_id)
    r = requests.post(url = 'http://API_address/querySql?token=' + token,  
                  headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                  data= 'sql=' + urllib.parse.quote(SQL) + '&format=csv_header') 
    s=str(r.content,'utf-8')
    channel_data = pd.read_csv(StringIO(s))
    
    # 去重複、轉數字(遺失值當作安卓)
    channel_data = channel_data.drop_duplicates(subset=['account_id'], keep='last')
    channel_data['initial_channel'] = channel_data['initial_channel'].apply(lambda x: 1 if x=="apple" else 0)
    
    #%%
    # 迴圈處理資料
    
    # 21日個別資料，篩選日期後將每日資料轉置
    def get_data_for_loop_fun(df, start_date, end_date, col_name):
        df_loop = df[(df['real_date'] >= start_date) & (df['real_date'] <= end_date)]
        df_loop = df_loop.assign(date = pd.to_datetime(end_date) + dt.timedelta(days=1))
        df_loop['days'] = 22 - (df_loop['date'] - pd.to_datetime(df_loop['real_date'])).dt.days
        df_loop = df_loop[['account_id', 'days', col_name]]
        
        # pivot轉置
        df_loop = df_loop.pivot(index='account_id', columns='days', values=col_name)
        df_loop.columns = [col_name + '_' + str(i) for i in range(1, 22)]
        df_loop = df_loop.assign(date = (pd.to_datetime(end_date) + dt.timedelta(days=1)).strftime('%Y-%m-%d') ).reset_index()
        
        return df_loop
    
    # 21日統整資料，篩選日期後計算次數
    def get_data_count_loop_fun(df, start_date, middle_date, middle_date_2, end_date, column):
        # 前7天跟後7天總和次數
        df_sum_loop1 = df[(df['real_date'] >= start_date) & (df['real_date'] < middle_date)]
        df_sum_loop1 = df_sum_loop1.groupby('account_id').count().reset_index().drop(['real_date'], axis = 1)
        df_sum_loop1.rename(columns={column : column+'_week_1'}, inplace=True)
        
        df_sum_loop2 = df[(df['real_date'] >= middle_date) & (df['real_date'] < middle_date_2)]
        df_sum_loop2 = df_sum_loop2.groupby('account_id').count().reset_index().drop(['real_date'], axis = 1)
        df_sum_loop2.rename(columns={column : column+'_week_2'}, inplace=True)
        
        df_sum_loop3 = df[(df['real_date'] >= middle_date_2) & (df['real_date'] <= end_date)]
        df_sum_loop3 = df_sum_loop3.groupby('account_id').count().reset_index().drop(['real_date'], axis = 1)
        df_sum_loop3.rename(columns={column : column+'_week_3'}, inplace=True)
        
        # outer join、加上日期、補NA值為0
        df_sum_loop = pd.merge(df_sum_loop1, df_sum_loop2, how='outer')
        df_sum_loop = pd.merge(df_sum_loop, df_sum_loop3, how='outer')
        df_sum_loop = df_sum_loop.assign(date = (pd.to_datetime(end_date) + dt.timedelta(days=1)).strftime('%Y-%m-%d') )
        df_sum_loop = df_sum_loop.fillna(0)
        
        return df_sum_loop
    
    # 21日統整資料，篩選日期後分別計算前後7天總和
    def get_data_sum_loop_fun(df, start_date, middle_date, middle_date_2, end_date, column):
        # 前7天跟後7天總和次數
        df_temp = df[['account_id', 'real_date', column]]
        
        df_sum_loop1 = df_temp[(df_temp['real_date'] >= start_date) & (df_temp['real_date'] < middle_date)]
        df_sum_loop1 = df_sum_loop1.groupby('account_id').sum().reset_index()
        df_sum_loop1.rename(columns={column : column+'_week_1'}, inplace=True)
        
        df_sum_loop2 = df_temp[(df_temp['real_date'] >= middle_date) & (df_temp['real_date'] < middle_date_2)]
        df_sum_loop2 = df_sum_loop2.groupby('account_id').sum().reset_index()
        df_sum_loop2.rename(columns={column : column+'_week_2'}, inplace=True)
        
        df_sum_loop3 = df_temp[(df_temp['real_date'] >= middle_date_2) & (df_temp['real_date'] <= end_date)]
        df_sum_loop3 = df_sum_loop3.groupby('account_id').sum().reset_index()
        df_sum_loop3.rename(columns={column : column+'_week_3'}, inplace=True)
        
        # outer join、加上日期、補NA值為0
        df_sum_loop = pd.merge(df_sum_loop1, df_sum_loop2, how='outer')
        df_sum_loop = pd.merge(df_sum_loop, df_sum_loop3, how='outer')
        df_sum_loop = df_sum_loop.assign(date = (pd.to_datetime(end_date) + dt.timedelta(days=1)).strftime('%Y-%m-%d') )
        df_sum_loop = df_sum_loop.fillna(0)
        
        return df_sum_loop
    
    # 創建空的dataFrame
    all_data = pd.DataFrame()
    
    for d in range(((dt.datetime.strptime(end_date, '%Y-%m-%d')) - dt.datetime.strptime(start_date, '%Y-%m-%d')).days + 1):
        # 設定日期 
        today = dt.datetime.strptime(start_date, '%Y-%m-%d') + dt.timedelta(days=d)
        today_str = today.strftime('%Y-%m-%d')
        start_date_loop = (today - dt.timedelta(days=21)).strftime('%Y-%m-%d')
        end_date_loop = (today - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        middle_date_loop = (today - dt.timedelta(days=14)).strftime('%Y-%m-%d')
        middle_date_loop_2 = (today - dt.timedelta(days=7)).strftime('%Y-%m-%d')
        
        start_date_lost = (today).strftime('%Y-%m-%d')
        end_date_lost = (today + dt.timedelta(days=6)).strftime('%Y-%m-%d')
        print(f'[{d}] 執行日期：{today_str}，抓取資料：[{start_date_loop} ~ {end_date_loop}]')
        
        # ==== 創角幾日 ====
        create_data_loop = create_data.copy()
        create_data_loop['date'] = today_str
        create_data_loop['create_days'] = (pd.to_datetime(create_data_loop['date']) - pd.to_datetime(create_data_loop['create_date'])).dt.days
        create_data_loop.drop(['create_date'], axis = 1, inplace = True)
        # 排除創角日小於0的資料
        create_data_loop = create_data_loop[create_data_loop['create_days'] >=0]
        
        # ==== 判斷流失 ====
        lost_data = online_data[(online_data['real_date'] >= start_date_lost) & 
                                (online_data['real_date'] <= end_date_lost)]
        lost_data['lost'] = 0
        lost_data = lost_data.drop(['real_date', 'online_time'], axis = 1).drop_duplicates()
        
        # 每日特徵，各資料取日期
        online_data_loop = get_data_for_loop_fun(online_data, start_date_loop, end_date_loop, 'online_time')
        first_hour_loop = get_data_for_loop_fun(first_hour, start_date_loop, end_date_loop, 'first_hour')
        game_ranked_time_loop = get_data_for_loop_fun(game_time, start_date_loop, end_date_loop, 'game_ranked_time')
        game_unranked_time_loop = get_data_for_loop_fun(game_time, start_date_loop, end_date_loop, 'game_unranked_time')
        game_ranked_next_time_loop = get_data_for_loop_fun(game_next_time, start_date_loop, end_date_loop, 'game_ranked_next_time')
        game_unranked_next_time_loop = get_data_for_loop_fun(game_next_time, start_date_loop, end_date_loop, 'game_unranked_next_time')
        game_ranked_counts_loop = get_data_for_loop_fun(game_counts, start_date_loop, end_date_loop, 'game_ranked_counts')
        game_unranked_counts_loop = get_data_for_loop_fun(game_counts, start_date_loop, end_date_loop, 'game_unranked_counts')
        level_loop = get_data_for_loop_fun(level, start_date_loop, end_date_loop, 'level')
        level_training_loop = get_data_for_loop_fun(level, start_date_loop, end_date_loop, 'level_training')
        quest_loop = get_data_for_loop_fun(quest, start_date_loop, end_date_loop, 'quest')
        
        # 每周次數特徵，各資料取日期 & 計算前7天、後7天次數
        login_counts_loop = get_data_count_loop_fun(login_counts, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'login_counts')
        development_talent_loop = get_data_count_loop_fun(development_talent, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'development_talent')
        
        # 每周總合特徵，各資料取日期 & 計算前7天、後7天總和
        iaa_count_loop = get_data_sum_loop_fun(iaa_count, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'iaa_daily_count')
        diamond_loop = get_data_sum_loop_fun(diamond, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'diamond_consume')    
        online_data_week = get_data_sum_loop_fun(online_data, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'online_time')
        game_ranked_next_time_week = get_data_sum_loop_fun(game_next_time, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'game_ranked_next_time')
        game_unranked_next_time_week = get_data_sum_loop_fun(game_next_time, start_date_loop, middle_date_loop, middle_date_loop_2, end_date_loop, 'game_unranked_next_time')
        
        ### 登入天數
        # 定義一個函數，將非零值替換為1
        def replace_non_zero_with_one(x):
            if isinstance(x, float):
                return 1 if x != 0.0 else 0.0
            else:
                return x
        
        # 對 DataFrame 中的每個元素應用上述函數
        login_days = online_data_loop.fillna(0)
        login_days = login_days.applymap(replace_non_zero_with_one)
        
        login_days['login_days_week_1'] = login_days[[f'online_time_{x}' for x in range(1, 8)]].sum(axis=1)
        login_days['login_days_week_2'] = login_days[[f'online_time_{x}' for x in range(8, 15)]].sum(axis=1)
        login_days['login_days_week_3'] = login_days[[f'online_time_{x}' for x in range(15, 22)]].sum(axis=1)
        login_days = login_days[['account_id', 'date', 'login_days_week_1', 'login_days_week_2', 'login_days_week_3']]
        
        # 資料整合
        merge_data = online_data_loop.merge(create_data_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(login_counts_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(game_ranked_time_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(game_unranked_time_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(game_ranked_counts_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(game_unranked_counts_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(development_talent_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(quest_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(iaa_count_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(diamond_loop, on = ['account_id', 'date'], how = 'left')\
            .merge(login_days, on = ['account_id', 'date'], how = 'left')\
            .merge(online_data_week, on = ['account_id', 'date'], how = 'left')\
            .merge(game_ranked_next_time_week, on = ['account_id', 'date'], how = 'left')\
            .merge(game_unranked_next_time_week, on = ['account_id', 'date'], how = 'left')\
            .merge(lost_data, on = ['account_id'], how = 'left')
        
        ### 遺失值處理
        # 遺失值補0，lost的遺失值補1
        merge_data['lost'] = merge_data['lost'].fillna(1)
        merge_data = merge_data.fillna(0)
        
        # next_time補值
        merge_data = merge_data.merge(game_ranked_next_time_loop, on = ['account_id', 'date'], how = 'left')\
        .merge(game_unranked_next_time_loop, on = ['account_id', 'date'], how = 'left')
        merge_data = merge_data.fillna(merge_data.max())
        
        # 不補遺失值的資料在這合併
        merge_data = merge_data.merge(first_hour_loop, on = ['account_id', 'date'], how = 'left')\
                     .merge(level_loop, on = ['account_id', 'date'], how = 'left')\
                     .merge(level_training_loop, on = ['account_id', 'date'], how = 'left')\
                     .merge(channel_data, on = ['account_id'], how = 'left')
        
        # 註冊未滿14天的玩家排除，不用來訓練模型、預測
        merge_data = merge_data[merge_data['create_days'] >= 21]
        
        # 合併資料、篩選第21天(預測前一天)需上線
        merge_data = merge_data[merge_data['online_time_21'] > 0]
        all_data = pd.concat([all_data, merge_data])
        
    return all_data
