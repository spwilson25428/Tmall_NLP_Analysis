import pandas as pd
import numpy as np
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pickle

pd.set_option('display.max_columns', 30, 'display.max_colwidth', 70, 'display.width', 100,
              'display.max_rows', 30)
plt.rc('font', family='SimHei', size='15')
# ===============================================================================================================
# Path Setting
path_training = r'D:\Haier\01 Project\Tmall Data Cluster\Model\情感语料库_汇总.xlsx'
path_stopword = r'D:\Haier\01 Project\Tmall Data Cluster\Model\stopwords-zh.txt'
path_testing = r'D:\Haier\01 Project\Tmall Data Cluster\Model\验证数据集.xlsx'
model_path = r'D:\Haier\01 Project\Tmall Data Cluster\Model\sentiment_model.h5'
tokenizer_path = r'D:\Haier\01 Project\Tmall Data Cluster\Model\tokenizer.pickle'
metadata_path = r'D:\Haier\01 Project\Tmall Data Cluster\Model\metadata.pickle'

# Stop words
stopwords = [i.rstrip('\n') for i in open(path_stopword, encoding='utf-8')]

# load train data
train_data = pd.read_excel(path_training, encoding='utf-8')
train_data['情感'] = np.where(train_data['情感'] == '正面', 'pos', 'neg')
train_data = train_data.astype({'评价内容': str})

# under-sampling and over-sampling the training data
train_data_pos = train_data[train_data['情感'] == 'pos']
train_data_neg = train_data[train_data['情感'] == 'neg']
train_data_pos = train_data_pos.sample(frac=0.1)
train_data_neg = train_data_neg.sample(n=len(train_data_pos), replace=True)
train_data = pd.concat([train_data_pos, train_data_neg], axis=0, ignore_index=True)

# convert training data to list of tuples
train_data = [i for i in train_data.itertuples(index=False, name=False)]

# tokenize
totalX = []
totalY = [str(review[1]) for review in train_data]
r1 = re.compile('[\u4e00-\u9fff]+')
for review in train_data:
    seg_list = jieba.cut(review[0], HMM=True)
    seg_list = [word for word in seg_list if word not in stopwords]  # remove stop words
    seg_list = [word for word in filter(r1.match, seg_list)]  # remove special symbols
    totalX.append(seg_list)

# Maximum Sentence Length
maxLength = 40  # we will only cover sentence up to 40 words

# Keras input tokenizer
totalX = [" ".join(wordslist) for wordslist in totalX]
input_tokenizer = Tokenizer(num_words=30000)
input_tokenizer.fit_on_texts(totalX)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))
vocab_size = len(input_tokenizer.word_index) + 1  # total vocabulary size

# Save input tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(input_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert output to [0, 1] array: 0 is positive, 1 is negative
target_tokenizer = Tokenizer(3)
target_tokenizer.fit_on_texts(totalY)
totalY = np.array(target_tokenizer.texts_to_sequences(totalY)) - 1
totalY = totalY.reshape(-1, )
totalY = to_categorical(totalY, num_classes=2)
output_dimen = totalY.shape[1]

# Save meta data
target_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
sentiment_tag = [target_index[1], target_index[2]]
metaData = {'maxLength': maxLength, 'vocab_size': vocab_size, 'output_dimen': output_dimen,
            'sentiment_tag': sentiment_tag}
with open(metadata_path, 'wb') as handle:
    pickle.dump(metaData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Build the model
embedding_size = 256
model = Sequential()
model.add(layer=Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=maxLength))
model.add(layer=GRU(units=256, dropout=0.1, return_sequences=True))
model.add(layer=GRU(units=256, dropout=0.1))
model.add(layer=Dense(units=32, activation='relu'))
model.add(layer=Dense(units=output_dimen, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
x_train, x_val, y_train, y_val = train_test_split(totalX, totalY, test_size=0.3)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=10, verbose=1,
                    validation_data=(x_val, y_val), callbacks=[es])


# Plot Model Accuracy
# fig, ax = plt.subplots()
# plt.plot(np.arange(1, 5), history.history['val_loss'])


# Define Prediction Function
def find_features(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    seg = jieba.cut(text, cut_all=False, HMM=True)
    seg = [word for word in seg if word not in stopwords]
    text = " ".join(seg)
    textarray = [text]
    textarray = np.array(pad_sequences(input_tokenizer.texts_to_sequences(textarray), maxlen=maxLength))
    return textarray


def predict_result(text):
    features = find_features(text)
    predicted = model.predict(features, verbose=1)[0]  # we have only one sentence to predict, so take index 0
    prob = predicted.max()
    prediction = sentiment_tag[predicted.argmax()]
    return prediction, prob


# predict the review data
test_data = pd.read_excel(path_testing, encoding='utf-8')
test_data = test_data.astype({'评价内容': str, '情感': str})
test_data['预测情感'] = test_data['评价内容'].apply(lambda x: predict_result(x)[0])
test_data['预测情感'] = np.where(test_data['预测情感'] == 'pos', '正面', '负面')
accuracy = accuracy_score(test_data['情感'], test_data['预测情感'])
labels = ['正面', '负面']
cm = confusion_matrix(test_data['情感'], test_data['预测情感'])
sns.heatmap(cm, annot=True, fmt='d')

# Save the Model
model.save(model_path)