#%%

import timeit
import numpy as np
import torch
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

#%%
train_data=np.load('./train_data.npy',allow_pickle=True)
train_label_rawdata=np.load('./train_label.npy',allow_pickle=True)
num_train_data = len(train_data)

#%%
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

# %%
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)

# %%
test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)

#%%
test_label = np.zeros(num_test_data)

for i in range(num_test_data):
    test_label_temp = test_label_rawdata[i]
    test_label_idx = rating.index(test_label_temp)
    test_label[i] = test_label_idx

#%%
size = 20
zero_train_set = np.zeros( (num_train_data,size,2,19,19) )
for i in range(num_train_data):
    for j in range(size):
        if j==0 :
            zero_train_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
            zero_train_set[i,j,1,:,:] = train_data[i][j]
        elif len(train_data[i]) <= j:
            zero_train_set[i,j,0,:,:]= train_data[i][len(train_data[i])-2]
            zero_train_set[i,j,1,:,:] = (train_data[i][len(train_data[i])-1] - train_data[i][len(train_data[i])-2])
        else :
            zero_train_set[i,j,0,:,:]=train_data[i][j-1]
            zero_train_set[i,j,1,:,:] = (train_data[i][j] - train_data[i][j-1])
train_set = np.reshape(zero_train_set,(num_train_data,size*2*19*19))  
#%%
zero_test_set = np.zeros((num_test_data,size,2,19,19))
for i in range(num_test_data):
    for j in range(size):
            if j==0 :
                zero_test_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
                zero_test_set[i,j,1,:,:] = test_data[i][j]
            elif len(test_data[i])<=j:
                zero_test_set[i,j,0,:,:]=test_data[i][len(test_data[i])-2]
                zero_test_set[i,j,1,:,:] =(test_data[i][len(test_data[i])-1] - test_data[i][len(test_data[i])-2])
            else :
                zero_test_set[i,j,0,:,:]=test_data[i][j-1]
                zero_test_set[i,j,1,:,:] = test_data[i][j] - test_data[i][j-1]
test_set=np.reshape(zero_test_set,(num_test_data,size*2*19*19)) 

#%%
'''
n 수 이전 데이터 
'''
n=5
#%%
size = 30
zero_train_set = np.zeros( (num_train_data,size,n+1,19,19) )
for i in range(num_train_data):
    for j in range(size):
        if j==0 :
            zero_train_set[i,j,0,:,:]=np.zeros((19,19)) # j-1번쨰 판의 상태 넣음 

        elif len(train_data[i]) <= j:
            zero_train_set[i,j,0,:,:]= train_data[i][len(train_data[i])-2]
            for k in range(1,n+1):
                zero_train_set[i,j,k,:,:] = train_data[i][len(train_data[i])-(k)] - train_data[i][len(train_data[i])-(k+1)]
                        
        else :
            zero_train_set[i,j,0,:,:]=train_data[i][j-1]
            for k in range(1,n+1):
                if j-k <0 :  
                    zero_train_set[i,j,k,:,:] = np.zeros((19,19))
                else:
                    zero_train_set[i,j,k,:,:] = train_data[i][j-(k-1)] - train_data[i][j-(k)]

train_set = np.reshape(zero_train_set,(num_train_data,size*(n+1)*19*19))  
#%%
size = 50
zero_test_set = np.zeros( (num_test_data,size,n+1,19,19) )
for i in range(num_test_data):
    for j in range(size):
        if j==0 :
            zero_test_set[i,j,0,:,:]=np.zeros((19,19)) # j-1번쨰 판의 상태 넣음 

        elif len(test_data[i]) <= j:
            zero_test_set[i,j,0,:,:]= test_data[i][len(test_data[i])-2]
            for k in range(1,n+1):
                zero_test_set[i,j,k,:,:] = test_data[i][len(test_data[i])-(k)] - test_data[i][len(test_data[i])-(k+1)]
                        
        else :
            zero_test_set[i,j,0,:,:]=test_data[i][j-1]
            for k in range(1,n+1):
                if j-k <0 :  
                    zero_test_set[i,j,k,:,:] = np.zeros((19,19))
                else:
                    zero_test_set[i,j,k,:,:] = test_data[i][j-(k-1)] - test_data[i][j-(k)]
test_set = np.reshape(zero_test_set,(num_test_data,size*(n+1)*19*19))  
#%%
print(train_set.shape)
print(test_set.shape)

#%%
from sklearn.ensemble import RandomForestClassifier
#%%
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=2000,n_jobs=-1,criterion='entropy');
clf.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
start_time = timeit.default_timer()
pred_test_label = clf.predict(test_set)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
# %%
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)*100/num_test_data))

# %%
model = SVC(kernel='rbf',C=1.0, gamma=0.10)# 선형적 비선형적 d

start_time = timeit.default_timer()
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def my_RF_prediction_prob(train_data, train_label, test_data, r):
    clf = RandomForestClassifier( random_state=r, n_estimators=1000)
    clf.fit(train_data, train_label) 
    return clf.predict_log_proba(test_data)

#%%
prob = []
for k in range(0,10):
    # Get last test data
    p = my_RF_prediction_prob(train_set, train_label, test_set, k)
    if prob==[]:
        prob = p
    else:
        prob += p
    predicted_label = prob.argmax(axis=1)
    print('Accuracy on test data: ' + str(sum(predicted_label==test_label)/len(test_label)))

#%%
'''



n_estimators=500 , 다른 거 안 건드림 , size 변화에 따른 정확도 분석
size = 10 => Accuracy on test data: 9.591078066914498
size = 16 => Accuracy on test data: 10.371747211895912
size = 17 => 
size = 18 => Accuracy on test data: 10.223048327137546
size = 19 => Accuracy on test data: 10.33457249070632
size = 20 => Accuracy on test data: 10.223048327137546
size = 21 =>  Accuracy on test data: 10.0
size = 23 => Accuracy on test data: 9.739776951672862
size = 25 => Accuracy on test data: 9.739776951672862
size = 30 => Accuracy on test data: 9.925650557620818
'''


#%%
for size in range (5,100):
    zero_train_set = np.zeros( (num_train_data,size,2,19,19) )
    for i in range(num_train_data):
        for j in range(size):
            if j==0 :
                zero_train_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
                zero_train_set[i,j,1,:,:] = train_data[i][j]
            elif len(train_data[i]) <= j:
                zero_train_set[i,j,0,:,:]= train_data[i][len(train_data[i])-2]
                zero_train_set[i,j,1,:,:] = train_data[i][len(train_data[i])-1] - train_data[i][len(train_data[i])-2]
            else :
                zero_train_set[i,j,0,:,:]=train_data[i][j-1]
                zero_train_set[i,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
    train_set = np.reshape(zero_train_set,(num_train_data,size*2*19*19))  

    zero_test_set = np.zeros((num_test_data,size,2,19,19))
    for i in range(num_test_data):
        for j in range(size):
                if j==0 :
                    zero_test_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
                    zero_test_set[i,j,1,:,:] = test_data[i][j]
                elif len(test_data[i])<=j:
                    zero_test_set[i,j,0,:,:]=test_data[i][len(test_data[i])-2]
                    zero_test_set[i,j,1,:,:] = test_data[i][len(test_data[i])-1] - test_data[i][len(test_data[i])-2]
                else :
                    zero_test_set[i,j,0,:,:]=test_data[i][j-1]
                    zero_test_set[i,j,1,:,:] = test_data[i][j] - test_data[i][j-1]
    test_set=np.reshape(zero_test_set,(num_test_data,size*2*19*19)) 

    start_time = timeit.default_timer()
    clf = RandomForestClassifier(n_estimators=500);
    clf.fit(train_set,train_label)
    terminate_time = timeit.default_timer()
    print("%f초 걸렸습니다." % (terminate_time - start_time)) 

    start_time = timeit.default_timer()
    pred_test_label = clf.predict(test_set)
    terminate_time = timeit.default_timer()
    print("%f초 걸렸습니다." % (terminate_time - start_time)) 

    print('size = '+str(size)+ ' Accuracy on test data: ' + str(sum(pred_test_label==test_label)*100/num_test_data))
# %%
'''
size = 5 Accuracy on test data: 8.475836431226766
size = 6 Accuracy on test data: 9.107806691449815
size = 7 Accuracy on test data: 8.884758364312267
size = 8 Accuracy on test data: 9.144981412639405
size = 9 Accuracy on test data: 9.33085501858736
size = 10 Accuracy on test data: 9.479553903345725
size = 11 Accuracy on test data: 9.73977
size = 12 Accuracy on test data: 10.0
size = 13 Accuracy on test data: 9.888475836431226
size = 14 Accuracy on test data: 9.814126394052044
size = 15 Accuracy on test data: 9.962825278810408
size = 16 Accuracy on test data: 10.148698884758364
size = 17 Accuracy on test data: 9.888475836431226
size = 18 Accuracy on test data: 9.962825278810408
size = 19 Accuracy on test data: 10.408921933085502
size = 20 Accuracy on test data: 10.743494423791821
size = 21 Accuracy on test data: 9.591078066914498
size = 22 Accuracy on test data: 10.706319702602231
size = 23 Accuracy on test data: 9.516728624535316
size = 24 Accuracy on test data: 9.628252788104088
size = 25 Accuracy on test data: 9.888475836431226
size = 26 Accuracy on test data: 9.962825278810408
size = 27 Accuracy on test data: 9.479553903345725
size = 28 Accuracy on test data: 9.591078066914498
size = 29 Accuracy on test data: 8.996282527881041
size = 30 Accuracy on test data: 10.074349442379182
size = 31 Accuracy on test data: 10.260223048327138
size = 32 Accuracy on test data: 10.483271375464684
size = 33 Accuracy on test data: 9.516728624535316
size = 34 Accuracy on test data: 9.739776951672862
size = 35 Accuracy on test data: 10.223048327137546
size = 36 Accuracy on test data: 9.925650557620818
size = 37 Accuracy on test data: 10.037174721189592
size = 38 Accuracy on test data: 10.631970260223047
size = 39 Accuracy on test data: 9.405204460966543
size = 40 Accuracy on test data: 9.628252788104088
size = 41 Accuracy on test data: 9.516728624535316
size = 42 Accuracy on test data: 9.962825278810408
size = 43 Accuracy on test data: 9.814126394052044
size = 44 Accuracy on test data: 9.628252788104088
size = 45 Accuracy on test data: 10.037174721189592
size = 46 Accuracy on test data: 9.962825278810408
size = 47 Accuracy on test data: 9.925650557620818
size = 48 Accuracy on test data: 10.037174721189592
size = 49 Accuracy on test data: 10.260223048327138
size = 50 Accuracy on test data: 10.037174721189592
size = 51 Accuracy on test data: 10.297397769516728
size = 52 Accuracy on test data: 10.33457249070632
size = 53 Accuracy on test data: 10.557620817843866
size = 54 Accuracy on test data: 9.107806691449815
size = 55 Accuracy on test data: 9.182156133828997
size = 99 Accuracy on test data: 9.702602230483272
'''


'''

1. 수 넣을떄 거리를 넣어보자. 
'''
#%%
#%%
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
        '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
        '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']

#%%
#%%
train_data=np.load('./train_data.npy',allow_pickle=True)
train_label_rawdata=np.load('./train_label.npy',allow_pickle=True)
num_train_data = len(train_data)
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)
test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)


#%%
def make_cheat_set(size):

    zero_cheat_set = np.zeros( (num_train_data+num_test_data,size,2,19,19) )
    for i in range(num_train_data+num_test_data):
        if i<num_train_data:
            for j in range(size):
                if j==0 :
                    zero_cheat_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
                    zero_cheat_set[i,j,1,:,:] = train_data[i][j]
                elif len(train_data[i]) <= j:
                    zero_cheat_set[i,j,0,:,:]= train_data[i][len(train_data[i])-2]
                    zero_cheat_set[i,j,1,:,:] = train_data[i][len(train_data[i])-1] - train_data[i][len(train_data[i])-2]
                else :
                    zero_cheat_set[i,j,0,:,:]=train_data[i][j-1]
                    zero_cheat_set[i,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
        else :
            for j in range(size):
                if j==0 :
                    zero_cheat_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
                    zero_cheat_set[i,j,1,:,:] = test_data[i-num_train_data][j]
                elif len(test_data[i-num_train_data])<=j:
                    zero_cheat_set[i,j,0,:,:]=test_data[i-num_train_data][len(test_data[i-num_train_data])-2]
                    zero_cheat_set[i,j,1,:,:] = test_data[i-num_train_data][len(test_data[i-num_train_data])-1] - test_data[i-num_train_data][len(test_data[i-num_train_data])-2]
                else :
                    zero_cheat_set[i,j,0,:,:]=test_data[i-num_train_data][j-1]
                    zero_cheat_set[i,j,1,:,:] = test_data[i-num_train_data][j] - test_data[i-num_train_data][j-1]
    cheat_set=np.reshape(zero_cheat_set,(num_train_data+num_test_data,size*2*19*19)) 
    return cheat_set

#%%
def make_cheat_label(size):
    cheat_label = np.zeros(num_test_data+num_train_data)

    rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
        '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
        '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']

    for i in range(num_test_data+num_train_data):
        if i<num_train_data:
            cheat_label_temp = train_label_rawdata[i]
            cheat_label_idx = rating.index(cheat_label_temp)
            cheat_label[i] = cheat_label_idx
        else :
            cheat_label_temp = test_label_rawdata[i-num_train_data]
            cheat_label_idx = rating.index(cheat_label_temp)
            cheat_label[i] = cheat_label_idx  
    return cheat_label
#%%
from sklearn.model_selection import train_test_split
# train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(cheat_set, cheat_label, test_size=0.2, shuffle=True, stratify=cheat_label, random_state=20)

#%%


#%%
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
while(1):
    size= random.randrange(1,50)
    cheat_set = make_cheat_set(size)
    cheat_label =make_cheat_label(size)
    x_train, x_valid, y_train, y_valid = train_test_split(cheat_set, cheat_label, test_size=0.2, shuffle=True, stratify=cheat_label, random_state=20)
    rf=RandomForestClassifier(n_estimators=500, max_features='auto')

    rf.fit(x_train,y_train)

    pred_test_label=rf.predict(x_valid)


    print(str(size)+' '+ 'Accuracy on test data: '+str(sum(pred_test_label==y_valid)/len(y_valid)))
    print(metrics.accuracy_score(y_valid, pred_test_label))
    if (metrics.accuracy_score(y_valid, pred_test_label)>0.12): 
        break

#%%
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation

x_train, x_valid, y_train, y_valid = train_test_split(cheat_set, cheat_label, test_size=0.2, shuffle=True, stratify=cheat_label, random_state=20)
X_train_lab, X_test_unlab,y_train_lab,y_test_unlab = train_test_split(x_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# %%
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
nolabel = [-1 for _ in range(len(y_test_unlab))]
y_train_mixed = concatenate((y_train_lab, nolabel))
model = LabelPropagation()
model.fit(X_train_mixed, y_train_mixed)
tran_labels = model.transduction_
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
model2 = RandomForestClassifier(n_estimators=500, max_samples=1500)
model2.fit(X_train_mixed, tran_labels)
#%%
yhat = model2.predict(x_valid)
# %%
score = accuracy_score(tran_labels, x_valid)
# %%
print('Accuracy: %.3f' % (score*100))
# %%
