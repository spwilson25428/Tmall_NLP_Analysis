import pandas as pd
import numpy as np
import jieba
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

pd.set_option('display.max_columns', 30, 'display.max_colwidth', 80, 'display.width', 100,
              'display.max_rows', 40)
plt.rc('font', family='SimHei', size='15')
# ===============================================================================================================
# Path Setting
path_review = r'dataset/评价信息.xlsx'
path_stopword = r'dataset/stopwords-zh.txt'
model_path = r'Model/sentiment_model.h5'
tokenizer_path = r'Model/tokenizer.pickle'
metadata_path = r'Model/metadata.pickle'

# Raw data loading
raw_data = pd.read_excel(path_review, encoding='utf-8')

# Stop words
stopwords = [i.rstrip('\n') for i in open(path_stopword, encoding='utf-8')]

# Select the useful columns
raw_data = raw_data[~raw_data['评论内容'].isin(['好评！', '好评', '好评!', 'good'])]  # remove meaningless comments
data = raw_data[['标题', '评论内容']]
data.columns = ['item', 'comment']

# Split items and comments sentence into words
data['item'] = data['item'].astype('str')  # convert data type
data['item_split'] = data['item'].apply(
    lambda x: [i for i in jieba.cut(x, HMM=True) if i not in stopwords])  # split item sentence
data['comment'] = data['comment'].astype('str')  # convert data type
data['comment_split'] = data['comment'].apply(
    lambda x: [i for i in jieba.cut(x, HMM=True) if i not in stopwords])  # split comment sentence
r = re.compile('[\u4e00-\u9fff]+')
data['comment_split'] = data['comment_split'].apply(
    lambda x: [i for i in filter(r.match, x)])  # remove symbols and digits, only keep Chinese

# Keep those comments have more than 2 words
data = data[data['comment_split'].map(lambda x: len(x) > 2)]
data.reset_index(drop=True, inplace=True)

# Sentiment Analysis ===============================================================================================
# Load Model
model = load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    input_tokenizer = pickle.load(handle)
with open(metadata_path, 'rb') as handle:
    metaData = pickle.load(handle)


# Define Prediction Function
def find_features(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    seg = jieba.cut(text, cut_all=False, HMM=True)
    seg = [word for word in seg if word not in stopwords]
    text = " ".join(seg)
    textarray = [text]
    textarray = np.array(pad_sequences(input_tokenizer.texts_to_sequences(textarray), maxlen=metaData['maxLength']))
    return textarray


def predict_result(text):
    features = find_features(text)
    predicted = model.predict(features, verbose=1)[0]  # we have only one sentence to predict, so take index 0
    prob = predicted.max()
    prediction = metaData['sentiment_tag'][predicted.argmax()]
    return prediction, prob


# predict the review data sentiment
data['sentiment'] = data['comment'].apply(lambda x: predict_result(x)[0])
data['sentiment'] = np.where(data['sentiment'] == 'pos', '正面', '负面')

# Categorize Vendor =================================================================================================
def get_item_vendor(sentence):
    if any(item in ['海尔'] for item in sentence):
        return '海尔'
    elif any(item in ['格力'] for item in sentence):
        return '格力'
    elif any(item in ['美的'] for item in sentence):
        return '美的'
    elif any(item in ['奥克斯'] for item in sentence):
        return '奥克斯'
    else:
        return '其他厂商'


data['vendor'] = data['item_split'].apply(get_item_vendor)

# Categorize item types =============================================================================================
def get_item_category(sentence):
    if any(item in ['冰箱', '雪柜', '电冰箱', '冰柜'] for item in sentence):
        return '冰箱'
    elif any(item in ['洗衣机', '洗衣', '电洗衣机'] for item in sentence):
        return '洗衣机'
    elif any(item in ['电视', '电视机', '液晶电视', '平板电视'] for item in sentence):
        return '电视机'
    elif any(item in ['空调', '空调机', '冷暖空调'] for item in sentence):
        return '空调'
    elif any(item in ['热水器', '电热水器'] for item in sentence):
        return '热水器'
    elif any(item in ['微波炉', '微波', '抽油烟机', '煤气炉', '吸油烟机', '燃气灶'] for item in sentence):
        return '厨电'
    else:
        return '其他电器'


data['item_category'] = data['item_split'].apply(get_item_category)
data.drop(columns='item_split', inplace=True)

# classify level 1 category for negative review data ================================================================
logistics_keyword = ['快递', '按时', '送货', '物流', '配送']
installation_keyword = ['调试', '安装人员', '指导', '安装师傅', '师傅', '上门', '安装', '安装工人']
repair_keyword = ['维修', '维修人员', '收费', '上门维修', '维修师傅', '退换', '退货', '换货', '更换']
customer_service_keyword = ['电话', '客服', '态度', '客服电话', '售后客服', '服务态度', '服务', '打电话', '售后']
marketing_keyword = ['用户体验', '体验', '页面', '发票', '价格', '降价', '赠品', '促销', '性价比', '优惠活动', '划算',
                     '售前']
quality_keyword = ['质量', '效果', '外观', '破损', '漂亮', '声音', '功能', '噪音', '容量', '制冷', '控制', '耐用',
                   '颜值', '操作', '好用', '味道', '毛病', '品质', '空间', '包装']


def get_comment_category_l1(sentence):
    """一级分类顺序： 物流-安装-维修-客服-营销-质量"""
    if any(item in logistics_keyword for item in sentence):
        return '物流'
    elif any(item in installation_keyword for item in sentence):
        return '安装'
    elif any(item in repair_keyword for item in sentence):
        return '维修'
    elif any(item in customer_service_keyword for item in sentence):
        return '客服'
    elif any(item in marketing_keyword for item in sentence):
        return '营销'
    elif any(item in quality_keyword for item in sentence):
        return '质量'
    else:
        return '其他'


data['l1_cat'] = data['comment_split'].apply(get_comment_category_l1)  # classify comments into level 1 category

# word frequency
comment_words = data.comment_split.apply(pd.Series).merge(data, left_index=True,
                                                          right_index=True)  # split the review words to individual columns
comment_words.drop(columns=['comment', 'comment_split', 'sentiment', 'item_category', 'l1_cat'], inplace=True)
comment_words = comment_words. \
    melt(id_vars=['item', 'vendor'], value_name="word"). \
    drop(columns='variable').dropna()  # melt all the words into one columns
comment_words = comment_words.groupby(['vendor', 'word'], as_index=False).count()
comment_words.to_csv('wordcloud.csv', index=None)

# Wordcloud Plot
# font = r'C:\Windows\Fonts\simhei.ttf'
# text = comment_words.word.values
# plt.figure()
# cloud_toxic = WordCloud(background_color='white',
#                         contour_color='steelblue',
#                         collocations=False,
#                         font_path=font,
#                         max_words=1000,
#                         width=2500, height=1800).generate(" ".join(text))
# plt.imshow(cloud_toxic)
# plt.axis('off')
# plt.tight_layout()

# classify level 2 category for negative review data ================================================================

# logistics data
data_logistics = data[data.l1_cat == '物流']
log_refund_kw = ['退钱', '退货', '退换', '换货', '赔偿', '售后', '换']  # 物流退换货
log_norm_kw = ['预约', '装卸', '签收', '暴力', '损坏', '磕碰', '划痕']  # 物流规范
log_speed_kw = ['延期', '及时', '时间']  # 物流速度
log_room_kw = ['上楼', '自提']  # 未送货入户
log_service_kw = ['收费', '态度', '服务', '素质', '投诉', '乱收费']  # 服务
log_install_kw = ['安装', '同步', '师傅']  # 送装不同步


def get_comment_category_l2_logistics(sentence):
    """二级分类顺序： 物流"""
    if any(item in log_refund_kw for item in sentence):
        return '退换货'
    elif any(item in log_norm_kw for item in sentence):
        return '物流规范'
    elif any(item in log_speed_kw for item in sentence):
        return '物流速度'
    elif any(item in log_room_kw for item in sentence):
        return '未送货入户'
    elif any(item in log_service_kw for item in sentence):
        return '服务规范'
    elif any(item in log_install_kw for item in sentence):
        return '送装不同步'
    else:
        return '其他'


data_logistics['l2_cat'] = data_logistics['comment_split'].apply(get_comment_category_l2_logistics)

# install data
data_install = data[data.l1_cat == '安装']
install_refund_kw = ['退钱', '退货', '退换', '换货', '赔偿', '售后', '换']  # 退换货
install_skill_kw = ['技能', '工艺', '检测']  # 安装技能
install_charge_kw = ['收费', '乱收费', '贵']  # 收费类
install_hardsell_kw = ['强推', '延保']  # 强卖增值
install_time_kw = ['及时', '慢', '不及时']  # 上门不及时
install_service_kw = ['服务', '态度', '规范', '投诉', '电话']  # 服务规范


def get_comment_category_l2_install(sentence):
    """二级分类顺序： 安装"""
    if any(item in install_refund_kw for item in sentence):
        return '退换货'
    elif any(item in install_skill_kw for item in sentence):
        return '安装技能'
    elif any(item in install_charge_kw for item in sentence):
        return '收费类'
    elif any(item in install_hardsell_kw for item in sentence):
        return '强卖增值'
    elif any(item in install_time_kw for item in sentence):
        return '上门不及时'
    elif any(item in install_service_kw for item in sentence):
        return '服务规范'
    else:
        return '其他'


data_install['l2_cat'] = data_install['comment_split'].apply(get_comment_category_l2_install)

# repair data
data_repair = data[data.l1_cat == '维修']
repair_refund_kw = ['退钱', '退货', '退换', '换货', '赔偿', '换']  # 退换货
repair_charge_kw = ['收费', '乱收费', '贵']  # 收费类
repair_time_kw = ['收费', '乱收费', '贵']  # 上门不及时
repair_service_kw = ['服务', '态度', '规范', '投诉', '电话', '客服']  # 服务规范


def get_comment_category_l2_repair(sentence):
    """二级分类顺序： 维修"""
    if any(item in repair_refund_kw for item in sentence):
        return '退换货'
    elif any(item in repair_charge_kw for item in sentence):
        return '收费类'
    elif any(item in repair_time_kw for item in sentence):
        return '上门不及时'
    elif any(item in repair_service_kw for item in sentence):
        return '服务规范'
    else:
        return '其他'


data_repair['l2_cat'] = data_repair['comment_split'].apply(get_comment_category_l2_repair)

# customer_service data
data_customer_service = data[data.l1_cat == '客服']
cs_phone_kw = ['电话', '不到位', '交流', '不通', '投诉', '举报', '拒接', '等待', '回复', '挂断']  # 话务客服
cs_service_kw = ['咨询', '态度', '服务', '不及时', '及时']  # 服务
cs_skill_kw = ['专业', '不够', '不懂', '没用', '不对']  # 技能
cs_refund_kw = ['退钱', '退货', '退换', '换货', '赔偿', '售后', '换']  # 退换货


def get_comment_category_l2_customer_service(sentence):
    """二级分类顺序： 客服"""
    if any(item in cs_phone_kw for item in sentence):
        return '话务客服'
    elif any(item in cs_skill_kw for item in sentence):
        return '技能'
    elif any(item in cs_refund_kw for item in sentence):
        return '退换货'
    elif any(item in cs_service_kw for item in sentence):
        return '服务规范'
    else:
        return '其他'


data_customer_service['l2_cat'] = data_customer_service['comment_split'].apply(
    get_comment_category_l2_customer_service)

# marketing data
data_marketing = data[data.l1_cat == '营销']
marketing_user_kw = ['活动', '执行', '购物', '流程', '不清晰', '不知道']  # 用户体验
marketing_interface_kw = ['误导', '用户', '页面', '不精细', '欺骗', '界面']  # 页面参数
marketing_receipt_kw = ['票货', '不统一', '发票', '未提供', '没有', '不一样']  # 发票
marketing_product_kw = ['价格', '保护期', '变价', '降价', '涨价', '买后']  # 服务规范


def get_comment_category_l2_marketing(sentence):
    """二级分类顺序： 营销"""
    if any(item in marketing_user_kw for item in sentence):
        return '用户体验'
    elif any(item in marketing_interface_kw for item in sentence):
        return '页面参数'
    elif any(item in marketing_receipt_kw for item in sentence):
        return '发票'
    elif any(item in marketing_product_kw for item in sentence):
        return '产品价格'
    else:
        return '其他'


data_marketing['l2_cat'] = data_marketing['comment_split'].apply(get_comment_category_l2_marketing)

# quality data
data_quality = data[data.l1_cat == '质量']
quality_experience_kw = ['缺', '附件', '说明书', '没有']  # 产品体验
quality_appearance_kw = ['外观', '开箱', '低级', '廉价', '老式']  # 产品外观
quality_function_kw = ['机器', '性能', '问题', '不能', '不灵', '运作']  # 产品性能


def get_comment_category_l2_quality(sentence):
    """二级分类顺序： 质量"""
    if any(item in quality_experience_kw for item in sentence):
        return '产品体验'
    elif any(item in quality_appearance_kw for item in sentence):
        return '产品外观'
    elif any(item in quality_function_kw for item in sentence):
        return '产品性能'
    else:
        return '其他'


data_quality['l2_cat'] = data_quality['comment_split'].apply(get_comment_category_l2_quality)

# Compile final negative data
data = pd.concat([data_install, data_repair, data_customer_service, data_logistics, data_marketing, data_quality],
                 ignore_index=True, axis=0)
data.drop(columns='comment_split', inplace=True)
data.columns = ['商品', '评论', '情感', '商家', '商品类别', '一级责任位', '二级责任位']
data.to_excel(r'D:\Haier\01_Project\Tmall Data Cluster\最终评价分类.xlsx', engine='xlsxwriter', index=None)

# Sentiment distribution by L1 category
sentiment_group = data.groupby(['一级责任位', '情感'], as_index=False).评论.count()
sentiment_ratio = pd.pivot_table(sentiment_group, '评论', '一级责任位', '情感')
sentiment_ratio = sentiment_ratio.assign(正面比例=sentiment_ratio['正面'] / sentiment_ratio.sum(axis=1),
                                         负面比例=sentiment_ratio['负面'] / sentiment_ratio.sum(axis=1))
sentiment_ratio.drop(columns=['正面', '负面'], inplace=True)
sentiment_ratio = sentiment_ratio.sort_values('正面比例', ascending=False)

# Sentiment Plot
# fig, ax = plt.subplots(figsize=(8, 5))
# sentiment_ratio.plot(kind='bar', ax=ax)
# plt.title('正面与负面评价数量')
# plt.xlabel('')
# fig.autofmt_xdate(rotation=0)
# plt.tight_layout()

# group by L1 and L2 category
negative_data = data[data['情感'] == '负面']
data_group_l1_l2 = negative_data.groupby(['一级责任位', '二级责任位'], as_index=False).评论.count()
data_group_l1_l2 = data_group_l1_l2.groupby('一级责任位')[['二级责任位', '评论']]. \
    apply(lambda x: x.sort_values('评论', ascending=False)).reset_index(level=0)
data_group_l1_l2 = data_group_l1_l2[data_group_l1_l2['二级责任位'] != '其他']

# fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(16, 8))
# ax_list = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2]]
# i = 0
# for l1 in data_group_l1_l2['一级责任位'].unique():
#     sns.barplot(x='二级责任位', y='评论', data=data_group_l1_l2[data_group_l1_l2.一级责任位 == l1], ax=ax_list[i])
#     plt.setp(ax_list[i].get_xticklabels(), rotation=20, ha='right')
#     ax_list[i].title.set_text(l1)
#     ax_list[i].set_xlabel('')
#     ax_list[i].set_ylabel('')
#     i += 1
# fig.suptitle('负面评论二级责任位分类')
# plt.tight_layout(rect=[0, 0, 1, 0.95])
