import pandas as pd
import numpy as np
from aip import AipNlp

pd.set_option('display.max_columns', 20, 'display.max_colwidth', 80, 'display.width', 120)
# ------------------------------------------------------------------------------------------

path_review = r'D:\Haier\01 Project\Tmall Data Cluster\评价信息.xlsx'
raw_data = pd.read_excel(path_review, encoding='utf-8')
data = raw_data[['评论内容']].rename(columns={'评论内容': 'comment'})
data = data.astype({'comment': 'str'})
data = data[~data['comment'].isin(['好评！', '好评', '好评!', 'good', '此用户未填写评价内容'])]  # remove meaningless comments
data = data[data['comment'].map(len) >= 2]
data = data[-290000:-190000]
data.reset_index(drop=True, inplace=True)

# Define Baidu AI Access Code
appID = '17271809'
APIkey = 'Ku1Gojepz6lKsQUt39Ft31Sr'
secretKey = 'SIOkdPIQgQOZDhfotZVVBWDWUQoR86gq'
model = AipNlp(appId=appID, apiKey=APIkey, secretKey=secretKey)

# Sentiment Analysis
final_result = pd.Series()
for i in range(9990):
    print('part {} start'.format(i))
    sample = data.comment[i*10:i*10+10]
    error_index = pd.Series([True])
    while error_index.any():
        try:
            result = sample.apply(lambda x: model.sentimentClassify(x))
            error_index = result.map(lambda x: 'error_code' in x.keys())
            output = result[~error_index]
            final_result = pd.concat([final_result, output])
            sample = sample[error_index]
        except (UnicodeEncodeError, UnicodeDecodeError, ValueError) as e:
            error_index = pd.Series([False])
            pass

# Compile result
Compiled_result = pd.DataFrame(final_result, columns=['result'])
Compiled_result['评价内容'] = Compiled_result['result'].apply(lambda x: x['text'])
Compiled_result['情感'] = Compiled_result['result'].apply(lambda x: x['items'][0]['positive_prob'])
Compiled_result['情感'] = np.where(Compiled_result['情感'] > 0.4, '正面', '负面')
Compiled_result.drop(columns=['result'], inplace=True)
Compiled_result.to_csv(r'D:\Haier\01 Project\Tmall Data Cluster\model\情感语料库_10.csv', encoding='utf-8_sig', index=False)

