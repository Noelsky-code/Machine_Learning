# C=1.0 , GAMMA= 0.1 
#Accuracy on test data: 0.09256505576208178
# pred에 30분 걸림

# C=10.0 , GAMMA=0.1
#Accuracy on test data: 0.09739776951672863

#RandomForestClassifier(min_samples_split=6, n_estimators=500, n_jobs=8)
#0.09702602230483272
#%%
import timeit
import numpy as np
import torch
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

train_data=np.load('./train_data.npy',allow_pickle=True)
train_label_rawdata=np.load('./train_label.npy',allow_pickle=True)
num_train_data = len(train_data)
#%%
for i in range(9):
  print(i)
#%%
zero_train_set = np.zeros( (num_train_data, 10,19,19) )
for i in range(num_train_data):
  for j in range(9):
    temp_data=train_data[i][j+1]-train_data[i][j]
    zero_train_set[i,j,:,:]=temp_data
  zero_train_set[i,0,:,:]=train_data[i][0]


train_set=np.reshape(zero_train_set,(num_train_data,10*19*19)) # 0.07323420074349442

# %%
# 고칠필요 x 
#train_lable에 0~26 저장
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
          '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
          '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']

#%%
train_label = np.zeros(num_train_data)
cnt=0
for i in range(num_train_data):
  train_label_temp = train_label_rawdata[i]
  train_label_idx = rating.index(train_label_temp)
  train_label[i] = train_label_idx
    
print(sum(train_label==0)) # number of '18k' 
print(sum(train_label==26)) # number of '9d'
#print(shape(train_label))
#%% 
#러닝하는 부분 
model = SVC(kernel='rbf',C=1.0, gamma=0.10)# 선형적 비선형적 d
model= OneVsOneClassifier(model) 
#model = LinearSVC()
#%%
#러닝. 
start_time = timeit.default_timer()
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
#모델 저장 
joblib.dump(model,'./svc_C=1.0_gamma=0.1.pkl')

#%%
#모델 로드 
model= joblib.load('svc_C=1.0_gamma=0.1.pkl')

# %%
#테스트 데이터 텐서를 만듬 
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)
print(num_test_data)

#%%

zero_test_set = np.zeros( (num_test_data, 10,19,19) )
for i in range(num_train_data):
  for j in range(9):
    temp_data=test_data[i][j+1]-test_data[i][j]
    zero_test_set[i,j,:,:]=temp_data
  zero_test_set[i,0,:,:]=test_data[i][0]

test_set=np.reshape(zero_test_set,(num_test_data,10*19*19))

#%%
# test data를 model에 돌린 결과 label 만들어짐 
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

#%%
#test label로 만들어주고 아래에서 비교. 

test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)
test_label = np.zeros(num_test_data)

for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label[i] = test_label_idx

len(test_label)
# %%
######

print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

#%%
# 교차검증 
from sklearn.model_selection import GridSearchCV

svm_clf= SVC(kernel='rbf',random_state=100)
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
             'gamma':[0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}
grid_svm = GridSearchCV(svm_clf, param_grid = parameters, cv = 5,n_jobs=4)
grid_svm.fit(train_set,train_label)

#%%

print(grid_svm.best_estimator_)
print(grid_svm.best_score_)
print(grid_svm.best_params_)


#####
######
# %%
start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=10.0, gamma=0.10)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

#%%
#모델 저장
joblib.dump(model,'./svc_C=10.0_gamma=0.1.pkl') 
#%%
#모델 로드
model= joblib.load('svc_C=10.0_gamma=0.1.pkl')
# %%
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

# %%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%
start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=1.0, gamma=0.10,random_state=100)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%

start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=1.0, gamma=0.10)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%
start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=9.0, gamma=0.10)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))


# %%
start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=10.0, gamma=0.11)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))



# %%
#0~9 말고 1~10 
two_train_set = np.zeros( (num_train_data, 10,19,19) )
for i in range(num_train_data):
  for j in range(10):
    temp_data=train_data[i][j+1]-train_data[i][j]
    zero_train_set[i,j,:,:]=temp_data

train_set=np.reshape(zero_train_set,(num_train_data,10*19*19))



model = SVC(kernel='rbf',C=10.0, gamma=0.10)
model.fit(train_set,train_label) 

#%%
two_test_set = np.zeros( (num_test_data, 10,19,19) )
for i in range(num_train_data):
  for j in range(10):
    temp_data=test_data[i][j+1]-test_data[i][j]
    two_test_set[i,j,:,:]=temp_data
  

test_set=np.reshape(two_test_set,(num_test_data,10*19*19))
# %%
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))


# %%
from sklearn.ensemble import RandomForestClassifier

train_set = np.reshape(zero_train_set,(num_train_data,10*19*19))
test_set= np.reshape(zero_test_set,(num_test_data,10*19*19))

rf_clf = RandomForestClassifier(n_estimators=500,min_samples_split=6,n_jobs=-1)
rf_clf.fit(train_set,train_label)
# %%
start_time = timeit.default_timer()
pred_test_label = rf_clf.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))
# %%
##
two_train_set = np.zeros( (num_train_data, 9,19,19) )
for i in range(num_train_data):
  for j in range(8):
    temp_data=train_data[i][j+1]-train_data[i][j]
    zero_train_set[i,j,:,:]=temp_data
  two_train_set[i,0,:,:]=train_data[i][0]

train_set_=np.reshape(two_train_set,(num_train_data,9*19*19))

two_test_set = np.zeros( (num_test_data, 9,19,19) )
for i in range(num_train_data):
  for j in range(8):
    temp_data=test_data[i][j+1]-test_data[i][j]
    two_test_set[i,j,:,:]=temp_data
  two_test_set[i,0,:,:]=test_data[i][0]
  
test_set_=np.reshape(two_test_set,(num_test_data,9*19*19))

rf_clf = RandomForestClassifier(n_estimators=1000,n_jobs=6)
rf_clf.fit(train_set_,train_label)
start_time = timeit.default_timer()
pred_test_label = rf_clf.predict(test_set_)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))
#%%
# %%
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [10, 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [2,4,6,8,12,16, 20]
            }
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(train_set,train_label)

# %%
grid_cv.best_params_
clf = RandomForestClassifier(max_depth=10,min_samples_leaf=12,min_samples_split=2,n_estimators=100);
clf.fit(train_set,train_label)
#{'max_depth': 10,
# 'min_samples_leaf': 12,
# 'min_samples_split': 2,
# 'n_estimators': 100}
# %%
start_time = timeit.default_timer()
pred_test_label = clf.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))
# %%
two_train_set = np.zeros( (num_train_data, 11,19,19) )
for i in range(num_train_data):
  for j in range(10):
    temp_data=train_data[i][j+1]-train_data[i][j]
    zero_train_set[i,j,:,:]=temp_data
  two_train_set[i,0,:,:]=train_data[i][0]

train_set_=np.reshape(two_train_set,(num_train_data,11*19*19))

two_test_set = np.zeros( (num_test_data, 11,19,19) )
for i in range(num_train_data):
  for j in range(10):
    temp_data=test_data[i][j+1]-test_data[i][j]
    two_test_set[i,j,:,:]=temp_data
  two_test_set[i,0,:,:]=test_data[i][0]
  
test_set_=np.reshape(two_test_set,(num_test_data,11*19*19))
# %%
clf.fit(train_set_,train_label)
start_time = timeit.default_timer()
pred_test_label = clf.predict(test_set_)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%
model = SVC(kernel='rbf',C=10.0, gamma=0.10)
model.fit(train_set_,train_label)

start_time = timeit.default_timer()
pred_test_label = model.predict(test_set_)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%
print(train_set[0])

# %%
