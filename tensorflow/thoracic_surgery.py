import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.getcwd();


#---------------------------------------------------
# 데이터을 로딩
df= pd.read_csv("thoracic_surgery.csv")
df.head()

# Converting Object dat types variables to integer data type
df.info()
df.head()

#---------------------------------------------------
# Boolean 데이터를 Integer로 변경
df[['pain', 
    'haemoptysis', 
    'dyspnoea', 
    'cough', 
    'weakness',  
    'diabetes',
    'mi_6months',
    'peri_artery_disease',
    'smoking',
    'asthma']] = np.where(
    df[['pain',
    'haemoptysis', 
    'dyspnoea', 
    'cough', 
    'weakness',  
    'diabetes',
    'mi_6months',
    'peri_artery_disease',
    'smoking',
    'asthma']].astype(str) == 'True', 1, 0)
df.head()

#---------------------------------------------------
# String 데이터를 Integer로 변경
df['icd10'] = df['icd10'].str[-1:].astype(int)
df['zubrod'] = df['zubrod'].str[-1:].astype(int)
df['tumor_size'] = df['tumor_size'].str[-1:].astype(int)
df.head()

data_set = df.values # 실제 데이터만 array 형태로 읽어온다
data_set

#---------------------------------------------------
# 데이터를 학습데이터, 테스트데이터로 분리
from sklearn.model_selection import train_test_split

train, test = train_test_split(data_set, test_size = 0.2) # 20%를 테스트 데이터로 사용
test.shape

#속성과 라벨(클래스) 분리하기
x_train = train[:, 0:16] # 0,1,2,3,...,15,16
y_train = train[:, 16]
x_train

#속성과 라벨(클래스) 분리하기
x_test = test[:, 0:16] # 0,1,2,3,...,15,16
y_test = test[:, 16]
y_test

from keras.models import Sequential
from keras.layers import Dense

# Sequential 객체, model 생성
model = Sequential()

# 2개의 층을 추가
model.add(Dense(30, input_dim = 16, activation='relu')) # activation 0, 1 판별함수, 1이면 다음 노드로 넘김
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='mean_squared_error', 
    optimizer='adam', 
    metrics=['accuracy'])# 역추적하는 함수로 adam을 사용

model.fit(
x_train, y_train, 
epochs=30, 
batch_size=10)

model.evaluate(x_test, y_test)

prediction = model.predict_classes(x_test)
prediction.ravel()

# 예측결과 출력
import pandas as pd

comparison = pd.DataFrame({'prediction':prediction.ravel(), 'ground_truth':y_test}).astype('int')
pd.set_option('display.max_rows', None)
comparison