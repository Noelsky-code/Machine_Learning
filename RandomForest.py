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
            zero_train_set[i,j,0,:,:]= np.zeros((19,19))
            zero_train_set[i,j,1,:,:] = np.zeros((19,19))
        else :
            zero_train_set[i,j,0,:,:]=train_data[i][j-1]
            zero_train_set[i,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
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
                zero_test_set[i,j,1,:,:] = test_data[i][len(test_data[i])-1] - test_data[i][len(test_data[i])-2]
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
size = 50
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
clf = RandomForestClassifier(n_estimators=2000,n_jobs=-1);
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