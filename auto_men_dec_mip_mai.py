import gspread
from oauth2client.service_account import ServiceAccountCredentials 
import pandas as pd
import numpy as np
from mip import Model, minimize, xsum
import os
import datetime
from icalendar import Calendar, Event
import pytz

def main(random_state = None, debug = False, path_input = None, dir_output = None, direct_in=False):
    if direct_in:
        json_keyfile_name="hoge/key_file_name.json"#Google API秘密鍵のパスを入力
        spreadsheet_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"#https://docs.google.com/spreadsheets/d/xxx/....のxxx部分を入力
        df_kagisime, df_gomisute, yyyymm=df_direct_input(json_keyfile_name, spreadsheet_key)
    else:
        path_input=input("Please enter path of schedule csv file:\n") if path_input is None else path_input
        
        df_kagisime, df_gomisute=df_input(path_input)
        fname = os.path.basename(path_input)
        key = ' - '
        yyyymm = fname[fname.find(key)+len(key):fname.find(' 予定表')]

    df_output, L_gomisute_members=member_decision_mip(df_kagisime=df_kagisime, df_gomisute=df_gomisute)

    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    if direct_in:
        update_spreadsheet(json_keyfile_name, spreadsheet_key, df_output)
    # .icsファイルを各メンバーごとに作成
    {make_ical(df_output, dir_output, yyyymm, member) for member in L_gomisute_members} # ゴミ捨てに登録されている全員のicsファイルを作成
    # 出力用ファイルの作成，出力
    df_output.to_csv(os.path.join(dir_output, yyyymm + ' 配置.csv'), encoding = 'utf_8_sig')

def df_direct_input(json_keyfile_name, spreadsheet_key):
    """Import schedule table directly from spread sheet.
    """
    #Set spreadsheet API and google drive API.
    SCOPES = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    #Set authentication info.
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, SCOPES)

    #Log in Google API using OAuth2 authentication information。
    gc =gspread.authorize(creds)

    #Open spreadsheet using spreadsheet key.
    SPREADSHEET_KEY = spreadsheet_key
    workbook = gc.open_by_key(SPREADSHEET_KEY)

    #Open worksheet of schedule table of next month.
    dt_now = datetime.datetime.now()
    intyyyymm=dt_now.year*100+dt_now.month+1 if dt_now.month!=12 else (dt_now.year+1)*100+1
    yyyymm=str(intyyyymm)
    worksheetname=yyyymm+" 予定表"
    worksheet = workbook.worksheet(worksheetname)

    cell_list = worksheet.get_all_values()
    df=pd.DataFrame(cell_list)
    df_input=df.drop(df.index[[0, 1]]).set_axis(df.iloc[1, :].tolist(), axis=1).reset_index(drop=True)
    index_date = [i for i, col in enumerate(df_input.columns) if '日付' in col]
    df_kagisime_input = df_input.iloc[:, index_date[0]:index_date[1]-1]
    df_gomisute_input = df_input.iloc[:, index_date[1]:index_date[2] if len(index_date) > 2 else None]

    df_kagisime=df_kagisime_input.copy()
    df_gomisute=df_gomisute_input.copy()
    for index in range(2, len(df_gomisute.columns.values)):
        if index < len(df_kagisime.columns.values):
            df_kagisime.iloc[:, index]=df_kagisime.iloc[:, index].apply(lambda x: x if x!='' else np.nan)
        df_gomisute.iloc[:, index]=df_gomisute.iloc[:, index].apply(lambda x: x if x!='' else np.nan)

    #Not assignable: False，Assignable: True
    df_gomisute.iloc[:, 2:]=df_gomisute.iloc[:, 2:].isnull()
    df_kagisime.iloc[:, 2:]=df_kagisime.iloc[:, 2:].isnull()

    df_gomisute['参加可能人数']=df_gomisute.iloc[:, 2:].sum(axis=1).astype(int)
    df_kagisime['参加可能人数']=df_kagisime.iloc[:, 2:].sum(axis=1).astype(int)
    df_gomisute['必要人数']=[4 if df_gomisute['参加可能人数'][i]>4 else df_gomisute['参加可能人数'][i] for i in df_gomisute.index.values]
    df_kagisime['必要人数']=[2 if df_kagisime['参加可能人数'][i]>2 else df_kagisime['参加可能人数'][i] for i in df_kagisime.index.values]

    return df_kagisime, df_gomisute, yyyymm

def df_input(input_path=None):
    """Original code: https://github.com/yu9824/member_decision main.py main function. 
    Partially changed for menber_decision_mip function.
    """
    df=pd.read_csv(input_path, skiprows=1)
    index_Unnamed = [i for i, col in enumerate(df.columns) if 'Unnamed: ' in col]
    df_kagisime_input = df.iloc[:, index_Unnamed[0]+1:index_Unnamed[1]]
    df_gomisute_input = df.iloc[:, index_Unnamed[1]+1:index_Unnamed[2] if len(index_Unnamed) > 2 else None]

    #割り当て不可→False，割り当て可能→True
    df_gomisute_input.iloc[:, 2:]=df_gomisute_input.iloc[:, 2:].isnull()
    df_kagisime_input.iloc[:, 2:]=df_kagisime_input.iloc[:, 2:].isnull() 
    #csv読み込みの際のカラム名重複回避の.1を除去
    df_gomisute=df_gomisute_input.rename(columns=lambda s: s.strip('.1'))
    df_kagisime=df_kagisime_input.copy()

    df_gomisute['参加可能人数']=df_gomisute.iloc[:, 2:].sum(axis=1).astype(int)
    df_kagisime['参加可能人数']=df_kagisime.iloc[:, 2:].sum(axis=1).astype(int)
    df_gomisute['必要人数']=[4 if df_gomisute['参加可能人数'][i]>4 else df_gomisute['参加可能人数'][i] for i in df_gomisute.index.values]
    df_kagisime['必要人数']=[2 if df_kagisime['参加可能人数'][i]>2 else df_kagisime['参加可能人数'][i] for i in df_kagisime.index.values]

    return df_kagisime, df_gomisute

def member_decision_mip(df_kagisime, df_gomisute):
    """Decide shift table by solving MIP using Python-MIP.
    See Python-MIP documentation(https://docs.python-mip.com/en/latest/index.html).
    """
    #Constant
    N_days=df_gomisute.shape[0]
    L_gactivedays=[i for i, v in enumerate(df_gomisute['参加可能人数']) if v!=0]
    L_kactivedays=[i for i, v in enumerate(df_kagisime['参加可能人数']) if v!=0]
    N_gactivedays=len(L_gactivedays)
    N_kactivedays=len(L_kactivedays)
    N_gomisute_members=df_gomisute.shape[1]-4
    N_kagisime_members=df_kagisime.shape[1]-4
    L_ksplited=[L_kactivedays[idx:idx + 5] for idx in range(0,N_kactivedays, 5)]
    N_weeks=len(L_ksplited)

    m=Model()

    #Variable for optimization.
    V_kshift_table = m.add_var_tensor((N_days,N_kagisime_members), name='V_kshift_table', var_type='INTEGER', lb=0, ub=1)
    z_kequal_person = m.add_var_tensor((N_kagisime_members, ), name='z_kequal_person', var_type='INTEGER')
    z_kequal_week =m.add_var_tensor((N_weeks,N_kagisime_members), name='z_kequal_week', var_type='INTEGER')

    V_gshift_table = m.add_var_tensor((N_days,N_gomisute_members), name='V_gshift_table', var_type='INTEGER', lb=0, ub=1)
    z_gequal_person = m.add_var_tensor((N_gomisute_members, ), name='z_gequal_person', var_type='INTEGER')

    z_sameday_kg = m.add_var_tensor((N_gactivedays, N_kagisime_members), name='z_kequal_person', var_type='INTEGER')

    C_equal_person=1000
    Cl_equal_week=[100 for i in range(N_kagisime_members)]
    Cl_sameday_kg=[10 for i in range(N_kagisime_members)]

    # 目的関数
    m.objective = minimize(
                        C_equal_person*xsum(z_kequal_person)
                        +C_equal_person*xsum(z_gequal_person)
                        +xsum(Cl_equal_week[i]*xsum(z_kequal_week[:, i]) for i in range(N_kagisime_members))
                        +xsum(Cl_sameday_kg[i]*xsum(z_sameday_kg[:, i])for i in range(N_kagisime_members))
                        )

    #制約条件：
    for i,r in df_kagisime.iloc[:, 2:].iterrows():
        #入れない日には入らない（入れない日=0(False)）, 必要人数を満たす
        for j in range(N_kagisime_members):
            m += V_kshift_table[i][j] <=r[j]
        m += xsum(V_kshift_table[i])==r['必要人数']

    for i,r in df_gomisute.iloc[:, 2:].iterrows():
        for j in range(N_gomisute_members):
            m += V_gshift_table[i][j] <=r[j]
        m += xsum(V_gshift_table[i])==r['必要人数']
    #絶対値に変換するための制限
    for i in range(N_kagisime_members):
        m += (xsum(V_kshift_table[:, i]) - (N_kactivedays*2)//N_kagisime_members) >=-z_kequal_person[i]
        m += (xsum(V_kshift_table[:, i]) - (N_kactivedays*2)//N_kagisime_members) <=z_kequal_person[i]
        for j, l_weekday in enumerate(L_ksplited):
            m += (xsum(V_kshift_table[l_weekday, i]) - (len(l_weekday)*2)//N_kagisime_members) >=-z_kequal_week[j, i]
            m += (xsum(V_kshift_table[l_weekday, i]) - (len(l_weekday)*2)//N_kagisime_members) <=z_kequal_week[j, i]
        #差をとって絶対値にする->最小化：(k, g)->z (1, 0), (0, 1)->1, (1, 1), (0, 0)->0, (2, 0)->2, (2, 1)->1
        for j, v in enumerate(L_gactivedays):
            m += (V_kshift_table[v, i]-V_gshift_table[v, i]) >=-z_sameday_kg[j, i]
            m += (V_kshift_table[v, i]-V_gshift_table[v, i])<=z_sameday_kg[j, i]
    for i in range(N_gomisute_members):
        m += (xsum(V_gshift_table[:, i]) - (N_gactivedays*4)//N_gomisute_members) >=-z_gequal_person[i]
        m += (xsum(V_gshift_table[:, i]) - (N_gactivedays*4)//N_gomisute_members) <=z_gequal_person[i]

    m.optimize()

    kagisime_shift_table=(V_kshift_table).astype(float).astype(int)
    gomisute_shift_table=(V_gshift_table).astype(float).astype(int)

    df_kagisime['Result'] = [', '.join(j for i,j in zip(r,df_kagisime.iloc[:, 2:2+N_kagisime_members].columns) if i==1) for r in kagisime_shift_table]
    df_gomisute['Result'] = [', '.join(j for i,j in zip(r,df_gomisute.iloc[:, 2:2+N_gomisute_members].columns) if i==1) for r in gomisute_shift_table]
    print('目的関数', m.objective_value)
    print(df_kagisime[['日付','曜日','Result']])
    print(df_gomisute[['日付','曜日','Result']])
    
    df_output=pd.DataFrame({'鍵閉め':list(df_kagisime['Result']), 'ゴミ捨て': list(df_gomisute['Result'])}, index=list(df_kagisime['日付']))
    L_gomisute_members=df_gomisute.iloc[:, 2:2+N_gomisute_members].columns

    return df_output, L_gomisute_members

def make_ical(df, dir_output, filename, member):
    """Make ical file of his shift table. Same as https://github.com/yu9824/member_decision main.py make_ical function.
    """
    # カレンダーオブジェクトの生成
    cal = Calendar()

    # カレンダーに必須の項目
    cal.add('prodid', 'yu-9824')
    cal.add('version', '2.0')

    # タイムゾーン
    tokyo = pytz.timezone('Asia/Tokyo')

    for name, series in df.iteritems():
        series_ = series[series.str.contains(member)]
        if name == '鍵閉め':
            start_td = datetime.timedelta(hours = 17, minutes = 45)   # 17時間45分
        elif name == 'ゴミ捨て':
            start_td = datetime.timedelta(hours = 14)   # 14時間
        else:
            continue
        need_td = datetime.timedelta(hours = 1)

        for date, cell in zip(series_.index, series_):
            # 予定の開始時間と終了時間を変数として得る．
            start_time = datetime.datetime.strptime(date, '%Y/%m/%d') + start_td
            end_time = start_time + need_td

            # Eventオブジェクトの生成
            event = Event()

            # 必要情報
            event.add('summary', name)  # 予定名
            event.add('dtstart', tokyo.localize(start_time))
            event.add('dtend', tokyo.localize(end_time))
            event.add('description', cell)  # 誰とやるかを説明欄に記述
            event.add('created', tokyo.localize(datetime.datetime.now()))    # いつ作ったのか．

            # カレンダーに追加
            cal.add_component(event)

    # カレンダーのファイルへの書き出し
    with open(os.path.join(dir_output, filename + member + '.ics'), mode = 'wb') as f:
        f.write(cal.to_ical())

def update_spreadsheet(json_keyfile_name, spreadsheet_key, df_output):
    """Update shift table of spreadsheet directly.
    Validate Google drive and spreadsheet API and set authentication info in advance. See "https://developers.google.com/workspace/guides/create-project".
    """
    #Set spreadsheet API and google drive API.
    SCOPES = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    #Set authentication info.
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, SCOPES)

    #Log in Google API using OAuth2 authentication information。
    gc =gspread.authorize(creds)

    #Open spreadsheet using spreadsheet key.
    SPREADSHEET_KEY = spreadsheet_key
    workbook = gc.open_by_key(SPREADSHEET_KEY)

    #Open worksheet of next month shift table.
    dt_now = datetime.datetime.now()
    intyyyymm=dt_now.year*100+dt_now.month+1 if dt_now.month!=12 else (dt_now.year+1)*100+1
    yyyymm=str(intyyyymm)
    outworksheetname=yyyymm+" 配置"
    outworksheet = workbook.worksheet(outworksheetname)

    #Find cell named '鍵閉め' and set cell range to copy df_output table.
    cell = outworksheet.find('鍵閉め')
    start_cell=num2alpha(cell.col)+str(cell.row+1)
    end_cell=num2alpha(cell.col+len(df_output.columns.values)-1)+str(cell.row+len(df_output.index.values))

    outworksheet.update(start_cell+':'+end_cell, df_output.values.tolist())

def num2alpha(num):
    """Convert number to alphabet. 
    Use for convert R1C1 style to A1 style in spreadsheet.
    """
    if num<=26:
        return chr(64+num)
    elif num%26==0:
        return num2alpha(num//26-1)+chr(90)
    else:
        return num2alpha(num//26)+chr(64+num%26)

if __name__ == '__main__':
    path_input="./example/GSS_test - 202111 予定表.csv"#Type path of schedule table csv file downloaded from spreadsheet unless direct_in=True.
    dir_output="./example"#Type path of output file(.ical, 予定表.csv).
    main(path_input=path_input, dir_output = dir_output, direct_in=False)