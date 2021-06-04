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
a = np.zeros((2,2))
a[0][0]=1
np.rot90(a,3)
print(a)
#np.rot90(train_data[0][0])
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
train_label = np.zeros(4*num_train_data)
cnt=0
for i in range(num_train_data):
    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    for j in range(4):
      train_label[cnt] = train_label_idx
      cnt = cnt + 1

#%%
size = 10
#%%

zero_train_set = np.zeros( (4*num_train_data,size,2,19,19) )
cnt = 0 
for i in range(num_train_data):
  for j in range(size):
    #print(train_data[i][j].shape)
    temp_90 = np.rot90(train_data[i][j][0],1)
    #print(temp_90.shape)
    temp_180 = np.rot90(train_data[i][j][0],2)
    temp_270 = np.rot90(train_data[i][j][0],3)
    if j==0 :
      zero_train_set[cnt,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
      zero_train_set[cnt,j,1,:,:] = train_data[i][j]
      zero_train_set[cnt+1,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
      zero_train_set[cnt+1,j,1,:,:] = temp_90
      zero_train_set[cnt+2,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
      zero_train_set[cnt+2,j,1,:,:] = temp_180
      zero_train_set[cnt+3,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
      zero_train_set[cnt+3,j,1,:,:] = temp_270    
      
    else :
      temp_90_ = np.rot90(train_data[i][j-1][0],1)
      temp_180_ = np.rot90(train_data[i][j-1][0],2)
      temp_270_ = np.rot90(train_data[i][j-1][0],3)
      zero_train_set[cnt,j,0,:,:]=train_data[i][j-1]
      zero_train_set[cnt,j,1,:,:] = train_data[i][j] - train_data[i][j-1]
      zero_train_set[cnt+1,j,0,:,:]=  temp_90_
      zero_train_set[cnt+1,j,1,:,:] = temp_90- temp_90_
      zero_train_set[cnt+2,j,0,:,:]=temp_180_
      zero_train_set[cnt+2,j,1,:,:] = temp_180- temp_180_
      zero_train_set[cnt+3,j,0,:,:]=temp_270_
      zero_train_set[cnt+3,j,1,:,:] = temp_270- temp_270_
  cnt = cnt + 4   
      
train_set=np.reshape(zero_train_set,(4*num_train_data,size*2*19*19))

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


#%%
from sklearn.ensemble import RandomForestClassifier
start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=500,max_depth=40,n_jobs=5,random_state=42);
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
test_label = np.zeros(num_test_data)

for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label[i] = test_label_idx

# %%

sum=0
for i in range(num_test_data):
    if pred_test_label[i]==test_label[i]:
        sum+=1
print(sum*100/num_test_data)
#%%
#print(type(test_label))
#print(type(pred_test_label))
#print(test_label==pred_test_label)
print(pred_test_label)
print(test_label)
print('Accuracy on test data: ' + str(sum(pred_test_label==test_label)*100/num_test_data))
# %%
