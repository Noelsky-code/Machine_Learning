
#%%
import timeit
import numpy as np
import torch
import joblib
import pickle
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier


train_data=np.load('./train_data.npy',allow_pickle=True)
train_label_rawdata=np.load('./train_label.npy',allow_pickle=True)

num_train_data = len(train_data)
print(len(train_data[1]))

#%%
all_data_len = 0
for i in range (num_train_data):
  all_data_len += len(train_data[i])
  print(i)
print(all_data_len)

#%%
#데이터 정리해야함 
start_time = timeit.default_timer() # 시작 시간 체크
last_train_data = np.zeros( (all_data_len, 19,19) )
cnt=0
for i in range (num_train_data):
  len_train_data = len(train_data[i])
  for j in range (len_train_data):
    temp_last_data = train_data[i][j]
    last_train_data[cnt, :, :] = temp_last_data
    cnt+=1

terminate_time = timeit.default_timer() # 종료 시간 체크 
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

# %%
#reshape 공부 
last_train_data = np.reshape(last_train_data,(all_data_len, 19*19))
print(last_train_data.shape) # check shape of last_train_data

# %%
# 고칠필요 x 
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
          '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
          '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']

#%%
#train_lable에 0~26 저장  
train_label = np.zeros(all_data_len)
cnt=0
for i in range(num_train_data):
  train_label_temp = train_label_rawdata[i]
  train_label_idx = rating.index(train_label_temp)
  for j in range(len(train_data[i])):
    train_label[cnt] = train_label_idx
    cnt+=1
    
print(sum(train_label==0)) # number of '18k' 
print(sum(train_label==26)) # number of '9d'
#print(shape(train_label))


#%% 
#러닝하는 부분 
model = LogisticRegression(solver='liblinear')# 선형적 비선형적 d
model= OneVsOneClassifier(model) 
#model = LinearSVC()
#%%
#러닝. 
start_time = timeit.default_timer()
model.fit(last_train_data,train_label)
print(len(model.estimators_)) #3
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 16분 
#%%
#모델 저장 
joblib.dump(model,'./logistic_baduk_model_logica.pkl')

#%%
#모델 로드 
model= joblib.load('logistic_baduk_model_logica.pkl')

# %%
#테스트 데이터 텐서를 만듬 
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)
print(num_test_data)
all_data_len = 0
for i in range (num_test_data):
  all_data_len += len(test_data[i])
print(all_data_len)

#%%

last_test_data = np.zeros( (all_data_len, 19,19) )
cnt=0
for i in range (num_test_data):
  len_test_data = len(test_data[i])
  for j in range (len_test_data):
    temp_last_data = test_data[i][j]
    last_test_data[cnt, :, :] = temp_last_data
    cnt+=1

last_test_data = np.reshape(last_test_data, (all_data_len, 19*19))

#%%
# test 하는곳 
####################################################
####################################################
temp_test_data= np.zeros((len(test_data[0]),19,19))
for i in range (len(test_data[0])):
  temp_test = test_data[0][i]
  temp_test_data[i, :, :] = temp_last_data

temp_test_data = np.reshape(temp_test_data,(len(test_data[0]),19*19))
start_time = timeit.default_timer()
pred_test_label = model.predict(temp_test_data)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
for i  in range (len(pred_test_label)):
  temp=pred_test_label[i]
  print(rating[temp])
  
print(rating[22])
test_label_rawdata = np.load('./test_label.npy', allow_pickle=True) 
print(test_label_rawdata[0])

####################################################
#%%
# test data를 model에 돌린 결과 label 만들어짐 

start_time = timeit.default_timer()
pred_test_label = model.predict(last_test_data)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
#최빈값 뽑아내기. 
len_pred = len(pred_test_label)
output = np.zeros(len(test_data))
print(type(pred_test_label))
cnt=0

for i in range (len(test_data)):
  a=np.zeros(len(test_data[i]))
  for j in range(0,len(test_data[i])) :
    a[j]=pred_test_label[cnt]
    cnt = cnt +1 
  a=a.astype(int)
  output[i]=np.bincount(a).argmax()
  print(output[i])


#%%
#test label로 만들어주고 아래에서 비교. 

test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)
test_label = np.zeros(num_test_data)

for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label[i] = test_label_idx

# %%
######
test_label=test_label.astype(int)
for i in range(0,1000):
    print(str(output[i])+" "+str(test_label[i]))
print('Accuracy on test data: ' + str(sum(output==test_label)/len(test_label)))

# %%
pred_train_data = model.predict(last_train_data)

# %%

#모델에 train_data 돌렷을때 결과 
train_label_a = np.zeros(num_train_data)
for i in range(num_train_data):
  train_label_temp = train_label_rawdata[i]
  train_label_idx = rating.index(train_label_temp)
  train_label_a[i] = train_label_idx


len_pred_train = len(pred_train_data)
output_train = np.zeros(len(train_data))

cnt=0

for i in range (len(train_data)):
  a=np.zeros(len(train_data[i]))
  for j in range(0,len(train_data[i])) :
    a[j]=pred_train_data[cnt]
    cnt = cnt +1 
  cnt=cnt
  a=a.astype(int)
  output_train[i]=np.bincount(a).argmax()
print('Accuracy on train data: ' + str(sum(output_train==train_label_a)/num_train_data))

#%%
#0~14번째만만 돌려보기
###
# 

#0~14번째만만 돌려보기...
#if k=0Accuracy on test data: 0.03382899628252788
#if k=1Accuracy on test data: 0.06802973977695168
#if k=2Accuracy on test data: 0.06505576208178439
#if k=3Accuracy on test data: 0.06617100371747212
#if k=4Accuracy on test data: 0.06431226765799257
#if k=5Accuracy on test data: 0.0654275092936803
#if k=6Accuracy on test data: 0.05947955390334572
#if k=7Accuracy on test data: 0.06505576208178439
#if k=8Accuracy on test data: 0.06319702602230483
#if k=9Accuracy on test data: 0.06171003717472119
#if k=10Accuracy on test data: 0.06728624535315986
#if k=11Accuracy on test data: 0.06468401486988848
#if k=12Accuracy on test data: 0.06319702602230483
#if k=13Accuracy on test data: 0.06356877323420074
# 
# 
#  
for k in range(0,14):
  test_data_len= len(test_data)
  temp_test_data = np.zeros( (test_data_len, 19,19))
  for i in range (test_data_len):
    temp_test_data[i, :, :] = test_data[i][k]

  temp_test_data = np.reshape(temp_test_data,(test_data_len,19*19))
  temp_test_label=model.predict(temp_test_data)
  print('if k=' +str(k) +'Accuracy on test data: ' + str(sum(temp_test_label==test_label)/len(test_label)))



