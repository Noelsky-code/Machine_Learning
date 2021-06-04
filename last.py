
#%%
import timeit
import numpy as np
import torch
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.datasets import make_classification

train_data=np.load('./train_data.npy',allow_pickle=True)
train_label_rawdata=np.load('./train_label.npy',allow_pickle=True)
num_train_data = len(train_data)
test_data=np.load('./test_data.npy',allow_pickle=True)
num_test_data=len(test_data)
test_label_rawdata = np.load('./test_label.npy', allow_pickle=True)


def my_RF_prediction_prob(train_data, train_label, test2_data, clf):
    return clf.predict_log_proba(test2_data)



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
size=20

#%%
test2_data=np.load('./Test2_data.npy',allow_pickle=True)
num_test2_data = len(test2_data)
zero_test2_set = np.zeros( (num_test2_data,size,2,19,19) )
for i in range(num_test2_data):
    for j in range(size):
        if j==0 :
            zero_test2_set[i,j,0,:,:]=np.zeros((19,19)) # j번쨰 판의 상태 넣음 
            zero_test2_set[i,j,1,:,:] = test2_data[i][j]
        elif len(test2_data[i]) <= j:
            zero_test2_set[i,j,0,:,:]= test2_data[i][len(test2_data[i])-2]
            zero_test2_set[i,j,1,:,:] = test2_data[i][len(test2_data[i])-1] - test2_data[i][len(test2_data[i])-2]
        else :
            zero_test2_set[i,j,0,:,:]=test2_data[i][j-1]
            zero_test2_set[i,j,1,:,:] = test2_data[i][j] - test2_data[i][j-1]
        
    test2_set=np.reshape(zero_test2_set,(num_test2_data,size*2*19*19)) 

#%%
cnt=0
while(1):
    while(1):
        cheat_set = make_cheat_set(size)
        cheat_label =make_cheat_label(size)
        x_train, x_valid, y_train, y_valid = train_test_split(cheat_set, cheat_label, test_size=0.2, shuffle=True, stratify=cheat_label, random_state=29)
        rf=RandomForestClassifier(n_estimators=2000,n_jobs=-1)
        rf.fit(x_train,y_train)
        pred_label=rf.predict(x_valid)


        print(str(size)+' '+ 'Accuracy on test data: '+str(sum(pred_label==y_valid)/len(y_valid)))
        print(metrics.accuracy_score(y_valid, pred_label))
        if (metrics.accuracy_score(y_valid, pred_label)>0.112): 
            break
    cnt= cnt+1

    prob = []
    p = my_RF_prediction_prob(cheat_set, cheat_label, test2_set,rf)
    if prob==[]:
        prob = p
    else:
        prob += p
    predicted_label = prob.argmax(axis=1)
    pred_test_label = predicted_label
    if cnt==10:
        break

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

    rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
            '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
            '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
    pred_test_label_txt = list_data = [str(rating[int(a)]).strip('\n\r') for a in pred_test_label]

    ans = pred_test_label_txt
    # 3. 암호화된 파일을 저장할 위치
    encrypt_ans_path = "./ans1.json"
    
    # 4. 암호화!(pycrytodome 설치)
    encrypt_data(key_path, ans, encrypt_ans_path)
# %%
