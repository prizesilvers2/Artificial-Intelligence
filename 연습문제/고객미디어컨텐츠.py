#!/usr/bin/env python
# coding: utf-8

# # <b>고객 미디어컨텐츠 선호장르 예측

# ## 1. Pandas 라이브러리 불러오기

# In[1]:


import pandas as pd


# ## 2. seaborn 라이브러리 설치 및 라이브러리 임포트

# In[2]:


get_ipython().system('pip install seaborn')
import seaborn as sns


# In[34]:


# 다음 문항을 풀기 전에 아래 코드를 실행해주세요

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# AIDU 내부 연동을 위한 라이브러리
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm
# AIDU와 연동을 위한 변수
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)

# tensorflow 관련 라이브러리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ## 3. pandas 라이브러리 함수로 데이터 프레임 읽어서 변수를 할당하는 코드를 넣으시오

# In[49]:


media = pd.read_csv(aidu_framework.config.data_dir + '/media_favor_genre_data.csv')


# In[50]:


# 데이터셋  확인
media.head()


# In[51]:


# 모든 컬럼의 데이터 타입 체크
media.dtypes


# ## 4. seaborn 라이브러리로 데이터 간의 상관관계를 히트맵 그래프로 나타내시오. 그래프에 상관관계 계수를 포함하세요.

# In[52]:


plt.figure(figsize=(16,8))

corr = media.corr()
sns.heatmap(corr , annot = True);


# ## 5. id 컬럼은 제거하고 sex_cd 컬럼값 중 under bar('_')를 X로 변경하시오
# ## 그리고 age_cd 데이터의 최빈값을 구하고 age_cd 컬럼의 under_bar를 aged_cd의 최빈값으로 변경하시오

# In[53]:


media.columns


# In[54]:


media.drop('id', inplace=True, axis=1)
media.sex_type_nm.replace('_', 'X', inplace = True)


# In[55]:


media.age_cd_nm.value_counts()
media.age_cd_nm.replace('_', 'A07', inplace = True)


# In[56]:


# 다음 문항을 풀기 전에 아래 코드를 실행해주세요.

media['favor_genre'].replace(['DRAMA', 'ACTION_SF', 'CHDR_FAMLY', 'CRIM_THRL', 'CMDY', 'ROMC_MELO', 'HORR', 'ANMT'],
                             [0,1,2,3,4,5,6,7], inplace= True)


# In[57]:


media.dtypes


# In[58]:


# 카테고리 변수를 더미 변수로 변환
cols = ['sex_type_nm', 'age_cd_nm']
dummies = pd.get_dummies(media[cols], drop_first = True)
dummies.head()


# In[59]:


media = media.drop(cols, axis =1)
media = pd.concat([media, dummies], axis =1)
media.head()


# In[60]:


media.info()


# In[61]:


# 추가로 데이터 처리 (원래는 안해줘도 됨)
media.cust_ctg_type_itg_cd.replace('_', 0, inplace = True)
media.cust_ctg_type_itg_cd.value_counts()


# ## 6. favor_genre 선호장르 데이터를 레이블 데이터 y로 할당하고, 나머지는 X로 할당하고 X와 y 데이터의 shape을 출력하세요

# In[62]:


y = media['favor_genre'].values
X = media.drop('favor_genre', axis =1).values
print(X.shape, y.shape)


# ## 7. scikit-learn train_test_split 함수를 이용하여 X, y 데이터로부터 훈련데이터셋과 검증 데이터셋을 70, 30 비율로 추출하시오. 데이터 추출시 y 데이터 분포와 유사하게 데이터를 추출하시오

# In[63]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X , y, test_size = 0.3, stratify = y)


# ## 8. 앙상블 중 랜덤 포레스트를 이용하여 고객의 미디어 선호장르를 분류하는 모델을 만들고 모델 성능을 출력하세요
# 
# RandomForestClassifier 파라미터 설정  
# n_estimators= 30, max_depth = 7, random_state = 21

# In[64]:


# 랜덤포레스트 분류 모델 사용
from sklearn.ensemble import RandomForestClassifier
# 모델 성능 분석
from sklearn.metrics import accuracy_score


# In[65]:


# 학습
rf = RandomForestClassifier(n_estimators= 30, max_depth = 7, random_state = 21)
rf.fit(X_train, y_train)


# In[66]:


# 모델 성능 구하기
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(accuracy)


# ## 9. 앞서 만든 데이터 셋을 이용하여, 딥러닝 모델 만들고 학습 정확도/손실, 검증정확도/손실 그래프로 표시하시오  
#   
# [학습모델 조건]  
# 1) 히든 레이어를 3객 이상으로 구성  
# 2) 과적합 방지하는 dropout 설정  
# 3) EarlyStopping 콜백으로 정해진 epoch 동안 모니터링 지표가 향상되지 않을 때, 훈련 중지하도록 설정  
# 4) ModelCheckpoint 콜백으로 validation performace 가 좋은 모델을 best_model.h5 파일로 저장  
#   
# [그래프 표시내용]  
# 학습 정확도 : acc  
# 학습 손실 : loss  
# 검증 정확도 : val_acc  
# 검증 손실 : val_loss  
#   
# + (필요시) X_train, X_valid 값을 0~1 사이의 값으로 스케일링 조정 => MinMaxScaler 이용  

# In[87]:


# 필요한 라이브러리 추가
# 1) 스케일링
from sklearn.preprocessing import MinMaxScaler

# 2) 딥러닝에 필요한 Keras 모델
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# categorical 라이브러리
from tensorflow.keras.utils import to_categorical


# In[69]:


# 스케일링 해주기(X_train, X_test)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[89]:


# 카테고라이징 해주기
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[90]:


# 내가 추가로 볼 것 
aa = pd.DataFrame(y_test)
aa.value_counts()
# y값이 여러개 이므로 categorical_crossentropy를 해주자!


# In[91]:


X_train.shape


# In[99]:


# 딥러닝 모델 구성

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (41,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation = 'softmax'))


# In[100]:


# 모델 내용 확인
model.summary()


# In[101]:


# 모델 최적화 컴파일링
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])


# In[102]:


# early_stopping, check_point
# verbose = 1 진행상황 출력 여부
early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
check_point = ModelCheckpoint('best_model.h5', verbose = 1, monitor='val_loss',
                              mode='min', save_best_only = True, save_weights_only = True)


# In[103]:


# 학습
epochs = 300
batch_size = 20

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, 
                    validation_data = (X_test, y_test), verbose =1, 
                    callbacks=[early_stopping, check_point])


# In[106]:


losses = pd.DataFrame(model.history.history)
losses.head()


# In[109]:


# 정확도(acc, loss 등)
plt.figure(figsize= (16,8))
plt.title('Accuracy')

plt.plot(losses['acc'], 'red', label='acc')
plt.plot(losses['loss'], 'blue', label='loss')
plt.plot(losses['val_acc'], 'yellow', label='val_acc')
plt.plot(losses['val_loss'], 'black', label='val_loss')

plt.xlabel('Epochs')
plt.ylabel('Acc')

plt.legend()
plt.show()


# In[ ]:




