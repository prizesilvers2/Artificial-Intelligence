#!/usr/bin/env python
# coding: utf-8

# # <b> 선박수주

# ## 1. Pandas 라이브러리를 넣어주세요. 

# In[96]:


import pandas as pd


# ## 2. Matplotlib를 plt로 넣어주세요.

# In[97]:


import matplotlib.pyplot as plt


# In[98]:


# 다음 문항을 풀기 전에 아래 코드를 실행하세요.
import numpy as np

get_ipython().system('pip install seaborn')
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model


# ## 3. 모델링을 위해 분석 및 처리할 데이터 파일을 읽어오시오. 

# In[99]:


# AIDU 내부 연동을 위한 라이브러리
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm
# AIDU와 연동을 위한 변수
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)

df = pd.read_csv(aidu_framework.config.data_dir + '/A0003MF.csv')


# In[100]:


df.head()


# ## 4. 데이터의 분포를 확인하려고 합니다. 데이터 컬럼의 분포를 확인하는 pairplot 그래프를 만드세요.  
#   
# 분포를 확인할 컬럼 : ['국제유가', '환율', '해운운임', '선박크기', '수주잔고']  
#   
# X 축에는 각각 컬럼 이름으로 표시하고 Y 축에는 입찰가로 표시하시오.

# In[101]:


sns.pairplot(data=df, x_vars =['국제유가', '환율', '해운운임', '선박크기', '수주잔고'], y_vars=['입찰가'] );


# In[102]:


# 수주여부 컬럼의 분포를 확인하는 histogram을 그리시오
# sns.histplot(data = df , x = '수주여부');
plt.hist(df['수주여부']);
plt.xlabel('Value')
plt.ylabel('Counts')
plt.title('Histogram Plot of Data')
plt.grid(True)
plt.show()


# In[103]:


# 환율과 국제 유가에 따른 수주여부 상관관계를 확인할 때, lmplot 그래프를 그리시오
sns.lmplot(data = df, x='환율', y='국제유가', hue = '수주여부');


# ## 5. 위 그래프에서 확인가능한 아웃라이어 데이터를 삭제하려고 합니다.   
#   
# ['선박크기', '수주잔고'] 컬럼의 아웃라이어 데이터를 삭제하세요.  
# - 선박크기 > 350000 인 대상을 삭제하세요.  
# - 수주잔고 > 250 인 대상을 삭제하세요.  

# In[104]:


condition = (df['선박크기'] > 350000) | (df['수주잔고'] >250)
df = df.drop(df[condition].index)


# ## 6. 수치형 데이터 간에 상관성을 보려고 합니다. 상관계수를 구해서 heatmap 그래프로 시각화하시오  
# 
# + annotation을 포함하세요.

# In[105]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.findSystemFonts(fontpaths=None, fontext='ttf')
#찾은 폰트를 기본 폰트로 설정하기
plt.rc('font', family='NanumGothicCoding')
plt.rc('axes', unicode_minus=False)


# In[106]:


get_ipython().system('apt-get install -y fonts-nanum')
get_ipython().system('fc-cache -fv')
get_ipython().system('rm ~/.cache/matplotlib -rf')


# In[107]:


corr = df.corr()
sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(corr, annot = True);


# ## 6. 모델링 성능을 제대로 얻기 위해서는 데이터 결측치를 처리해야합니다. 결측치를 처리하시오.

# In[108]:


df.info()


# In[109]:


df['선종'].fillna(0, inplace = True)
df['해운운임'].fillna(0, inplace = True)
df['선박크기'].fillna(0, inplace = True)
df.isnull().sum()

# df['선종'].replace(np.nan,0, inplace = True)


# ## 7. 범주형 데이터를 수치형 데이터로 변환해주는 데이터 전처리 방법 중 하나로 label encoder를 사용한다. Scikit-learn의 label encoder를 사용하여 범주형 데이터를 수치형 데이터로 변환하시오  
#   
# + 전처리 대상 컬럼 : '선주사' , '선종'  
# + fit_transform 을 활용하세요.  

# In[110]:


df['선종'].value_counts()
df['선종'].replace(0, '0',inplace = True)


# In[111]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['선주사'] = le.fit_transform(df['선주사'])
df['선종'] = le.fit_transform(df['선종'])
df.head()


# ## 8. 원 핫 인코딩은 범주형 변수를 1과 0 의 이진형 벡터로 변환하기 위하여 사용하는 방법입니다. 아래 조건에 해당하는 컬럼 데이터를 변환하세요.  
#   
# + 원핫 인코딩 대상 : object 타입의 컬럼  
# Pandas 의 get_dummies 함수를 활용하고 drop_first 는 'True' 파라 미터를 추가하세요.

# In[112]:


df.info()


# In[113]:


cols = ['유사선박수주경험','중국입찰여부','국내경쟁사입찰여부', '수주여부']
dummies = pd.get_dummies(df[cols], drop_first = True)
df = df.drop(cols, axis =1)
df = pd.concat([df, dummies], axis =1)
df.head()


# ## 9. 훈련과 검증에 사용할 데이터 셋을 분리하시오. '수주여부' 컬럼은 Label(y)로 할당한 후, 훈련데이터와 검증데이터셋으로 분리하시오. 데이터의 비율은 훈련데이터셋 70, 검증데이터셋 30 비율로 분리하고, y 데이터를 원래 데이터의 분포와 유사하게 추출되도록 옵션을 적용하세요.

# In[114]:


X = df.drop(['수주여부_yes'], axis =1)
y = df['수주여부_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)


# ## 10. 데이터 정규화를 StandardScaler를 사용하여 아래 조건에 따라 정규분포화, 표준화 하세요.  
#   
# Scikit-learn의 StandardScaler를 사용하세요.
# train set은 정규분포화(fit_transform) 하세요.
# test set은 표준화(transform)를 하세요.

# In[115]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()

X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# ## 11. 랜덤 포레스트 모델로 학습을 진행하세요.  
# 
# 결정트리의 개수는 100개  
# 최대 feature 개수는 9    
# 트리의 최대 깊이는 15  
# random_state 는 42로 설정하세요.  

# In[116]:


from sklearn.ensemble import RandomForestClassifier

fr= RandomForestClassifier(n_estimators=100, max_features=9, max_depth=15, random_state=42)


# In[117]:


# 학습
fr.fit(X_train, y_train)

# 점수
fr.score(X_test, y_test)


# ## 12. 위 모델의 성능을 평가하시오.  
#   
# y 값을 예측하여 confusion matrix를 구하고 heatmap그래프로 시각화하세요.  
# 또한 Scikit-learn의 classification report 기능을 사용하여 성능을 출력하세요.  
# 
# (= 동일한 말)  
# 11번 모델에서 만든 모델로 y값을 예측하여 y_pred에 저장하세요.  
# Confusion_matrix를 구하고 heatmap 그래프로 시각화하세요. 이때 annotation을 포함시키세요.  
# Scikit-learn의 classification report기능을 사용하여 클래스 별 precision, recall , f1-score를 출력하세요.  

# In[118]:


# 성능을 평가하기 위해 필요한 라이브러리 추가
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred = fr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.set(rc={'figure.figsize':(5,5)})
sns.heatmap(cm, annot = True);


# In[119]:


# classification report
print(classification_report(y_test, y_pred, target_names = ['class 0', 'class 1']))


# ## 13. 선박의 수주 여부를 예측하는 딥러닝 모델을 만들어라.  
#   
# [조건]
# 1) 히든레이어 3개이상으로 모델 구성, 과적합 장지 dropout을 설정  
# 2) Earlystopping 콜백으로 정해진 epoch동안 모니터링 지표가 향상되지 않을 떄 훈련을 중지하도록 설정  
# 3) ModelCheckpoint 콜백으로 validation performance가 좋은 모델을 '본인 핸드폰번호.h5' 파일로 저장  

# In[120]:


# # 필요한 라이브러리 추가
# from tensorflow.keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# In[128]:


batch_size = 2
epochs= 5

es = EarlyStopping(monitor ='val_acc', patience = 4, mode = 'max', verbose = 1)
mc = ModelCheckpoint('본인핸드폰번호.h5', monitor='val_acc', save_weights_only=True, save_best_only=True)

model = Sequential()
model.add(Dense(32, activation = 'relu', input_shape=(10,)))
model.add(Dropout(0.3))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics =['acc'])

history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,
                   callbacks= [es, mc], validation_data = (X_test, y_test))


# ## 14. 위 딥러닝 모델의 성능을 평가하려고 합니다.  
# 학습 정확도/손실, 검증 정확도/손실 그래프로 표시하시오.  
#   
# - 1개 그래프에 4가지 모두 표시하시오.
# - 범례를 'acc', 'loss', 'val_acc', 'val_loss'로 표시하세요.  
# - 타이틀은 'Accuracy'로 표시하세요.  
# - x축에는 Epochs라고 표시하고 Y축에는 Acc라고 표시하세요.

# In[130]:


losses = pd.DataFrame(model.history.history)

plt.figure(figsize= (16,8))
plt.plot(losses['acc'], 'red', label='acc')
plt.plot(losses['loss'], 'blue', label='loss')
plt.plot(losses['val_acc'], 'yellow', label='val_acc')
plt.plot(losses['val_loss'], 'black', label='val_loss')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.ylabel('Acc')
plt.show()


# In[ ]:




