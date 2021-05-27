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
            zero_train_set[i,j,1,:,:] = train_data[i][len(train_data[i])-1] - train_data[i][len(train_data[i])-2]
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
from sklearn.ensemble import RandomForestClassifier
#%%
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=500);
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
size = 20 = > 
'''