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

# %%
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)

# %%
test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)
test_label = np.zeros(num_test_data)

for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label[i] = test_label_idx
#%%
import math
def dist(a,b):
    y1 = np.where(np.logical_or(a==1,a==-1))[1][0]
    x1 = np.where(np.logical_or(a==1,a==-1))[2][0]
    y2 = np.where(np.logical_or(b==1,b==-1))[1][0]
    x2 = np.where(np.logical_or(b==1,b==-1))[2][0]
    return math.sqrt( abs(y2-y1)*abs(y2-y1) + abs(x2-x1)*abs(x2-x1))
    #return abs(y2-y1)+abs(x2-x1)
#%%
# 거리 평균값 확인하기 
avg = np.zeros(len(rating))
for i in range (num_train_data):
    sum =0
    for j in range(14):
        if j == 0 :
            sum = sum + dist(train_data[i][0],train_data[i][j+1]-train_data[i][j])
        else :
            sum = sum + dist(train_data[i][j]-train_data[i][j-1],train_data[i][j+1]-train_data[i][j])
    idx = (int)(train_label[i])
    avg[idx]+=sum
#%%
print(sum(train_label==0)) # number of '18k' 
for i in range(len(rating)):
    print(sum(train_label==0)) # number of '18k' 
    #sum((train_label==0))
    #print(div)
    #avg[i] = avg[i]/div
 #3번쨰 수 

# %%
zero_train_set = np.zeros( (num_train_data, 10,2,19,19) )
for i in range(num_train_data):
    for j in range(10):
        if j <4 : 
            zero_train_set[i,j,0,:,:]=train_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]
        else : 
            zero_train_set[i,j,0,:,:]=train_data[i][j]-train_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]

train_set=np.reshape(zero_train_set,(num_train_data,10*2*19*19)) 
#%%
model = SVC(kernel='rbf',C=10.0, gamma=0.01)# 선형적 비선형적 d

start_time = timeit.default_timer()
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

#%%
zero_test_set = np.zeros( (num_test_data, 10,2,19,19) )
for i in range(num_test_data):
    for j in range(10):
        if j <4 : 
            zero_test_set[i,j,0,:,:]=test_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]
        else : 
            zero_test_set[i,j,0,:,:]=test_data[i][j]-test_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]

#%%
test_set=np.reshape(zero_test_set,(num_test_data,10*2*19*19))
#%%
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))
#%%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))

# %%
'''
3개 
Accuracy on test data: 0.06951672862453531 -> 1~11 에 대해 앞선 3수의 상태 + 다음 수 
Accuracy on test data: 0.07323420074349442 -> 1~ 10 에 대해 '' 
Accuracy on test data: 0.07657992565055761 -> 1~9 에 대해 
Accuracy on test data: 0.08401486988847584 -> 1-8
Accuracy on test data: 0.08773234200743495 -> 1-7 
Accuracy on test data: 0.08996282527881042 -> 1-6 
Accuracy on test data: 0.08884758364312267 -> 1-5

2개 

1~10 : Accuracy on test data: 0.08475836431226766
'''

#%%
train_label = np.zeros(num_train_data*10)
cnt=0
for i in range(num_train_data):
    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    for j in range(10):
        train_label[i*10+j] = train_label_idx
# %%
zero_train_set = np.zeros( (num_train_data,10,2,19,19) )
for i in range(num_train_data):
    for j in range(10):
        if j <4 : 
            zero_train_set[i,j,0,:,:]=train_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]
        else : 
            zero_train_set[i,j,0,:,:]=train_data[i][j]-train_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]

train_set=np.reshape(zero_train_set,(num_train_data*10,2*19*19)) 
#%%
print(train_set.shape)
print(train_label.shape)
#%%
model = SVC(kernel='rbf',C=1.0, gamma=0.10)# 선형적 비선형적 d

start_time = timeit.default_timer()
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

#%%
zero_test_set = np.zeros( (num_test_data, 10,2,19,19) )
for i in range(num_test_data):
    for j in range(10):
        if j <4 : 
            zero_test_set[i,j,0,:,:]=test_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]
        else : 
            zero_test_set[i,j,0,:,:]=test_data[i][j]-test_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]

#%%
test_set=np.reshape(zero_test_set,(num_test_data*10,2*19*19))
#%%
test_label = np.zeros(num_test_data*10)
cnt=0
for i in range(num_test_data):
    test_label_temp = test_label_rawdata[i]
    test_label_idx = rating.index(test_label_temp)
    for j in range(10):
        test_label[i*10+j] = test_label_idx


#%%
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))
#%%
print(pred_test_label.shape)
#%%
len_pred = len(pred_test_label)
output = np.zeros(len_pred)
cnt=0

for i in range (len_pred):
    a=np.zeros(10)
    for j in range(0,10) :
        a[j]=pred_test_label[cnt]
    a=a.astype(int)
    output[i]=np.bincount(a).argmax()
    print(output[i])

#%%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)/len(test_label)))
# %%

zero_train_set = np.zeros( (num_train_data,7,2,19,19) )
for i in range(num_train_data):
    for j in range(7):
        if j <4 : 
            zero_train_set[i,j,0,:,:]=train_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]
        else : 
            zero_train_set[i,j,0,:,:]=train_data[i][j]-train_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[i,j,k+1,:,:] = train_data[i][j+k+1] - train_data[i][j+k]

train_set=np.reshape(zero_train_set,(num_train_data*7,2*19*19)) 
#%%
# 모든 데이터 넣기
all_train_data = 0 
for i in range (num_train_data):
    if len(train_data[i])>60 : 
        all_train_data +=60
    else :
        all_train_data +=len(train_data[i])-1
   
    
print(all_train_data)

zero_train_set = np.zeros((all_train_data,2,19,19))
cnt = 0 
for i in range(num_train_data):
    a = 0
    if len(train_data[i])>60 : 
        a=60
    else :
        a = len(train_data[i])-1
    
    for j in range (a):
        #print(cnt)
        if j==0:
            zero_train_set[cnt,0,:,:]=np.zeros((19,19))
            zero_train_set[cnt,1,:,:] = train_data[i][0]
        elif j <5 : 
            zero_train_set[cnt,0,:,:]=train_data[i][j-1] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[cnt,1,:,:] = train_data[i][j+k] - train_data[i][j+k-1]
        else : 
            zero_train_set[cnt,0,:,:]=train_data[i][j-1]-train_data[i][j-5] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_train_set[cnt,1,:,:] = train_data[i][j+k] - train_data[i][j+k-1]
        cnt= cnt+1
print(cnt)
train_set = np.reshape(zero_train_set,(all_train_data,(2*19*19)))
print(train_set.shape)

all_test_data = 0 
for i in range (num_test_data):
    if len(test_data[i])>60 : 
        all_test_data +=60
    else :
        all_test_data +=len(test_data[i])-1
   
    
print(all_test_data)

zero_test_set = np.zeros((all_test_data,2,19,19))
cnt = 0 
for i in range(num_test_data):
    a = 0
    if len(test_data[i])>60 : 
        a=60
    else :
        a = len(test_data[i])-1
    
    for j in range (a):
        #print(cnt)
        if j==0:
            zero_test_set[cnt,0,:,:]=np.zeros((19,19))
            zero_test_set[cnt,1,:,:] = test_data[i][0]
        if j <5 : 
            zero_test_set[cnt,0,:,:]=test_data[i][j-1] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[cnt,1,:,:] = test_data[i][j+k] - test_data[i][j+k-1]
        else : 
            zero_test_set[cnt,0,:,:]=test_data[i][j-1]-test_data[i][j-5] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[cnt,1,:,:] = test_data[i][j] - test_data[i][j+k-1]
        cnt= cnt+1
print(cnt)
test_set = np.reshape(zero_test_set,(all_test_data,(2*19*19)))
print(test_set.shape)
#%%
# 수정버전 
all_train_data = 0 
for i in range (num_train_data):
    if len(train_data[i])>70 : 
        all_train_data +=70
    else :
        all_train_data +=len(train_data[i])-1
   
    
print(all_train_data)

zero_train_set = np.zeros((all_train_data,2,19,19))
cnt = 0 
for i in range(num_train_data):
    a = 0
    if len(train_data[i])>70 : 
        a=70
    else :
        a = len(train_data[i])-1
    
    for j in range (a):
        if j==0 :
            zero_train_set[cnt,0,:,:]=np.zeros((19,19))
        else:
            zero_train_set[cnt,0,:,:]=train_data[i][j-1] # j번쨰 판의 상태 넣음 
        zero_train_set[cnt,1,:,:]= train_data[i][j] - train_data[i][j-1]
        cnt= cnt+1
print(cnt)
train_set = np.reshape(zero_train_set,(all_train_data,(2*19*19)))
print(train_set.shape)

all_test_data = 0 
for i in range (num_test_data):
    if len(test_data[i])>70 : 
        all_test_data +=70
    else :
        all_test_data +=len(test_data[i])-1
   
    
print(all_test_data)

zero_test_set = np.zeros((all_test_data,2,19,19))
cnt = 0 
for i in range(num_test_data):
    a = 0
    if len(test_data[i])>70 : 
        a=70
    else :
        a = len(test_data[i])-1
    
    for j in range (a):
        if j==0 :
            zero_test_set[cnt,0,:,:]=np.zeros((19,19))
        else:
            zero_test_set[cnt,0,:,:]=test_data[i][j-1] # j번쨰 판의 상태 넣음 
        zero_test_set[cnt,1,:,:]= test_data[i][j] - test_data[i][j-1]
        cnt= cnt+1

test_set = np.reshape(zero_test_set,(all_test_data,(2*19*19)))
print(test_set.shape)
#%%
zero_test_set = np.zeros( (num_test_data, 10,2,19,19) )
for i in range(num_test_data):
    for j in range(10):
        if j <4 : 
            zero_test_set[i,j,0,:,:]=test_data[i][j] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]
        else : 
            zero_test_set[i,j,0,:,:]=test_data[i][j]-test_data[i][j-4] # j번쨰 판의 상태 넣음 
            for k in range(1):
                zero_test_set[i,j,k+1,:,:] = test_data[i][j+k+1] - test_data[i][j+k]


test_set=np.reshape(zero_test_set,(num_test_data*10,2*19*19))

#%%
train_label = np.zeros(all_train_data)
cnt=0
for i in range(num_train_data):
    a = 0
    if len(train_data[i])>70 : 
        a=70
    else :
        a = len(train_data[i])-1

    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    for j in range (a):
        train_label[cnt] = train_label_idx
        cnt=cnt+1

#  랜덤 포레스트 
#%%
from sklearn.ensemble import RandomForestClassifier
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=200,min_samples_split= 6,min_samples_leaf=6,max_depth=17,n_jobs=5,random_state=42);
clf.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
# %%
print(test_set.shape)
start_time = timeit.default_timer()
pred_test_label = clf.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
clf.get_params()
#%%
from collections import Counter

pred_len = len(pred_test_label)
pred = np.zeros(num_test_data)
k=0
for i in range (num_test_data):
    
    if len(test_data[i])>70 : 
        a=70
    else :
        a = len(test_data[i])-1
    counter  = np.zeros(a)
    for j in range (a) :
        counter[j] = pred_test_label[k]
        k=k+1
    cnt = Counter(counter)
    pred[i] = cnt.most_common(1)[0][0]


#%%
#print(sum(pred==1.))
print('Accuracy on test data: ' + str(sum(pred==test_label)*100/len(test_label)))
# %%
#%%
train_label = np.zeros(num_train_data*10)
cnt=0
for i in range(num_train_data):
    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    for j in range(10):
        train_label[i*10+j] = train_label_idx

#%%
size = 11
# %%
zero_train_set = np.zeros( (num_train_data,size,2,19,19) )
for i in range(num_train_data):
    
    for j in range(size):
        if j==0 :
            zero_train_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
        else :
            zero_train_set[i,j,0,:,:]=train_data[i][j-1]
        zero_train_set[i,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
        

train_set=np.reshape(zero_train_set,(num_train_data,size*2*19*19)) 

#%%

zero_test_set = np.zeros( (num_test_data,size,2,19,19) )
for i in range(num_test_data):
    for j in range(size):
        if j==0 :
            zero_test_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
        else :
            zero_test_set[i,j,0,:,:]=test_data[i][j-1]
        zero_test_set[i,j,1,:,:] = test_data[i][j] - test_data[i][j-1]
        

test_set=np.reshape(zero_test_set,(num_test_data,size*2*19*19)) 

# %%
train_label = np.zeros(num_train_data)
cnt=0
for i in range(num_train_data):
  train_label_temp = train_label_rawdata[i]
  train_label_idx = rating.index(train_label_temp)
  train_label[i] = train_label_idx
#%%
from sklearn.ensemble import RandomForestClassifier
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=200,max_depth=17,n_jobs=5,random_state=42);
clf.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
# %%
print(test_set.shape)
start_time = timeit.default_timer()
pred_test_label = clf.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 


test_label = np.zeros(num_test_data)

for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label[i] = test_label_idx

# %%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)*100/len(test_label)))
# %%
'''
10번쨰 수 까지 -> depth = 15 -> 
'''
#%%
model = SVC(kernel='rbf',C=1.0, gamma=0.10)# 선형적 비선형적 d

start_time = timeit.default_timer()
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
start_time = timeit.default_timer()
pred_test_label = model.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))
# %%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)*100/len(test_label)))


# %%
#주의 시간 엄청오래걸림 
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [ 50,100,200,300],
           'max_depth' : [6, 8, 10, 12,14,16,18,20,22,24],
           'min_samples_leaf' : [4,6,8,10,12],
           'min_samples_split' : [2,4,6,8,12,16,20]
            }
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(train_set,train_label)
#%%
grid_cv.best_params_
#%%
size = 11 
zero_cheat_set = np.zeros( (num_train_data+num_test_data,size,2,19,19) )
for i in range(num_train_data+num_test_data):
    if i<num_train_data:
        for j in range(size):
            if j==0 :
                zero_cheat_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
            else :
                zero_cheat_set[i,j,0,:,:]=train_data[i][j-1]
            zero_cheat_set[i,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
    else :
        for j in range(size):
            if j==0 :
                zero_cheat_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
            else :
                zero_cheat_set[i,j,0,:,:]=test_data[i-num_train_data][j-1]
            zero_cheat_set[i,j,1,:,:] = test_data[i-num_train_data][j] - test_data[i-num_train_data][j-1]
cheat_set=np.reshape(zero_cheat_set,(num_train_data+num_test_data,size*2*19*19)) 
#%%
cheat_label = np.zeros(num_test_data+num_train_data)

for i in range(num_test_data+num_train_data):
    if i<num_train_data:
        cheat_label_temp = train_label_rawdata[i]
        cheat_label_idx = rating.index(cheat_label_temp)
        cheat_label[i] = cheat_label_idx
    else :
        cheat_label_temp = test_label_rawdata[i-num_train_data]
        cheat_label_idx = rating.index(cheat_label_temp)
        cheat_label[i] = cheat_label_idx  

# %%
#주의 시간 엄청오래걸림 
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [ 50,100,200,300],
           'max_depth' : [10,11,12,13,14,15,16,17,18,20,22,24]
            }
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(cheat_set,cheat_label)


# %%
from sklearn.ensemble import RandomForestClassifier
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=200,max_depth=17,n_jobs=5,random_state=42);
clf.fit(cheat_set,cheat_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
# %%
#주의 시간 엄청오래걸림 
from sklearn.model_selection import GridSearchCV

start_time = timeit.default_timer()
params = { 'n_estimators' : [300],
           'max_depth' : [17,25,35]
            }
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(cheat_set,cheat_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
grid_cv.best_params_
#%%
clf = RandomForestClassifier(n_estimators=300,max_depth=17,n_jobs=5,random_state=42);
clf.fit(cheat_set,cheat_label)
'''


'''
#%%

import json
from base64 import b64encode
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad
import timeit
import numpy as np
import torch
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

def read_txt(fileName):
    with open(fileName, 'rt') as f:
        list_data = [a.strip('\n\r') for a in f.readlines()]
    return list_data

def write_json(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_key(key_path):
    with open(key_path, "rb") as f:
        key = f.read()
    return key

def encrypt_data(key_path, ans_list, encrypt_store_path='ans.json'):
    key = load_key(key_path)
    data = " ".join([str(i) for i in ans_list])
    encode_data = data.encode()
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(encode_data, AES.block_size))
    iv = b64encode(cipher.iv).decode('utf-8')
    ct = b64encode(ct_bytes).decode('utf-8')
    write_json(encrypt_store_path, {'iv':iv, 'ciphertext':ct})

if __name__=="__main__":
    # 1.이메일을 통해서 전달 받은 키 파일의 경로 입력
    key_path = "./team12.pem"
    # 2. 예측한 결과를 텍스트 파일로 저장했을 경우 리스트로 다시 불러오기
    # 본인이 원하는 방식으로 리스트 형태로 예측 값을 불러오기만 하면 됨(순서를 지킬것)
    #raw_ans_path = "ans.txt" 
    #ans = read_txt(raw_ans_path)
  
    test_data=np.load('./Test2_data.npy',allow_pickle=True)
    num_test_data = len(test_data)
    zero_test_set = np.zeros( (num_test_data,size,2,19,19) )
    for i in range(num_test_data):
        for j in range(size):
            if j==0 :
                zero_test_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
            else :
                zero_test_set[i,j,0,:,:]=test_data[i][j-1]
            zero_test_set[i,j,1,:,:] = test_data[i][j] - test_data[i][j-1]
        

    test_set=np.reshape(zero_test_set,(num_test_data,size*2*19*19)) 
    pred_test_label = clf.predict(test_set)
      
#%%   
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
          '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
          '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
pred_test_label_txt = list_data = [str(rating[int(a)]).strip('\n\r') for a in pred_test_label]
print(pred_test_label_txt)

#%%
    ans = pred_test_label_txt
    # 3. 암호화된 파일을 저장할 위치
    encrypt_ans_path = "./ans1.json"
    
    # 4. 암호화!(pycrytodome 설치)
    encrypt_data(key_path, ans, encrypt_ans_path)
# %%
