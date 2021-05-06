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
train_data2=np.load('./test_data.npy',allow_pickle=True)
train_label_rawdata2=np.load('./test_label.npy',allow_pickle=True)
num_train_data = len(train_data)
num_train_data2 = len(train_data2)

#%%
num_cheat_data = num_train_data + num_train_data2
zero_train_set = np.zeros( (num_train_data+num_train_data2, 10,19,19) )
cnt=0
for i in range(num_train_data+num_train_data2):
    if i < num_train_data:
        for j in range(9):
            temp_data=train_data[i][j+1]-train_data[i][j]
            zero_train_set[i,j,:,:]=temp_data
        zero_train_set[i,0,:,:]=train_data[i][0]
    else:
        for j in range(9):
            temp_data=train_data2[cnt][j+1]-train_data2[cnt][j]
            zero_train_set[i,j,:,:]=temp_data
        zero_train_set[i,0,:,:]=train_data2[cnt][0]
        cnt=cnt+1

train_set=np.reshape(zero_train_set,(num_cheat_data,10*19*19))
print(train_set.shape)
# %%
#train_lable에 0~26 저장
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
        '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
        '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
#%%
train_label = np.zeros(num_cheat_data)
cnt=0
for i in range(num_cheat_data):
    if i< num_train_data:
        train_label_temp = train_label_rawdata[i]
        train_label_idx = rating.index(train_label_temp)
        train_label[i] = train_label_idx
    else :
        train_label_temp = train_label_rawdata2[cnt]
        train_label_idx = rating.index(train_label_temp)
        train_label[i] = train_label_idx
        cnt=cnt+1
# %%
print(sum(train_label==0)) # number of '18k' 
print(sum(train_label==26)) # number of '9d'
# %%

start_time = timeit.default_timer()
model = SVC(kernel='rbf',C=10.0, gamma=0.10)
model.fit(train_set,train_label)
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 


# %%
#모델 저장
joblib.dump(model,'./cheat_svc_C=10.0_gamma=0.1.pkl') 

#%%
# 모델 로드 
model= joblib.load('cheat_svc_C=10.0_gamma=0.1.pkl')


#%%
# 결과 나오는곳 

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


    model= joblib.load('cheat_svc_C=10.0_gamma=0.1.pkl')
    test_data=np.load('./Test2_data.npy',allow_pickle=True)
    num_test_data = len(test_data)
    zero_test_set = np.zeros( (num_test_data, 10,19,19) )

    for i in range(num_test_data):
        for j in range(9):
            temp_data=test_data[i][j+1]-test_data[i][j]
            zero_test_set[i,j,:,:]=temp_data
        zero_test_set[i,0,:,:]=test_data[i][0]

    test_set=np.reshape(zero_test_set,(num_test_data,10*19*19))
    pred_test_label = model.predict(test_set)

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
from sklearn.model_selection import GridSearchCV

svm_clf= SVC(kernel='rbf',random_state=100)
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
             'gamma':[0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}
grid_svm = GridSearchCV(svm_clf, param_grid = parameters, cv = 5,n_jobs=4)
grid_svm.fit(train_set,train_label)

print(grid_svm.best_estimator_)
print(grid_svm.best_score_)
print(grid_svm.best_params_)