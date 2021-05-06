#%%
#로드 
import numpy as np
import os
import torch
import torch.nn as nn
import timeit




train_data_path = "./train_data.npy"
train_label_path = "./train_label.npy"

train_data = np.load(train_data_path, allow_pickle=True)
train_label_rawdata = np.load(train_label_path, allow_pickle=True)


#%%
#레이팅 
rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
        '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
        '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
#%%
num_train_data = len(train_data)
train_label = np.zeros(num_train_data * 10)
for i in range(num_train_data):
    for j in range(0,10):
        train_label_temp = train_label_rawdata[i]
        train_label_idx = rating.index(train_label_temp)
        train_label[i*10 + j] = train_label_idx

#%%
#1~10판 기보 -> zero_test_set

zero_train_set = np.zeros( (num_train_data, 10,19,19) )
for i in range(num_train_data):
    for j in range(9):
        temp_data=train_data[i][j+1]-train_data[i][j]
        zero_train_set[i,j,:,:]=temp_data
    zero_train_set[i,0,:,:]=train_data[i][0]

#%%
X = torch.tensor(zero_train_set)
X = X.view(-1, 19*19)
print(X[-1])

#%%
Y = torch.tensor(train_label)
Y = Y.view(-1,1)
# %%
#학습파트 
start_time = timeit.default_timer()

device = torch.device('cuda')
X = X.to(device)
Y = Y.to(device)
model = nn.Sequential(
        nn.Linear(19*19, 381, bias=True),
        nn.ELU(),
        nn.Linear(381, 381, bias=True),
        nn.ELU(),
        nn.Linear(381, 381, bias=True),
        nn.ELU(),
        nn.Linear(381, 190, bias=True),
        nn.ELU(),
        nn.Linear(190, 95, bias=True),
        nn.ELU(),
        nn.Linear(95, 45, bias=True),
        nn.ELU(),
        nn.Linear(45, 1, bias=True),
        ).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5001): #5000번 학습
    optimizer.zero_grad()
    # forward 연산
    hypothesis = model(X.float())

    # 비용 함수
    cost = criterion(hypothesis, Y.float())
    cost.backward()
    optimizer.step()

    # 에포크마다 코스트 출력
    #if epoch % 1 == 0:
        #print(epoch, cost.item())

terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time)) 



# %%
print(X.shape)
print(Y.shape)

print(num_train_data)



#%%
# test 데이터 로드 
test_data_path = "./test_data.npy"
test_data = np.load(test_data_path, allow_pickle=True)


#%%
num_test_data = len(test_data)
zero_test_set = np.zeros( (num_test_data, 10,19,19) )
for i in range(num_test_data):
    for j in range(9):
        temp_data=test_data[i][j+1]-test_data[i][j]
        zero_test_set[i,j,:,:]=temp_data
    zero_test_set[i,0,:,:]=test_data[i][0]
#%%
print(num_test_data)
print(len(zero_test_set))



#%%
test_data_path = "./test_label.npy"
test_label_rawdata =np.load(test_label_path, allow_pickle=True)
#%%
test_label = np.zeros(num_test_data)
for i in range(num_test_data):
    test_label_temp = test_label_rawdata[i]
    test_label_idx = rating.index(test_label_temp)
    test_label[i] = test_label_idx
#%%
#텐서 사이즈 변경 
print(len(zero_test_set))
X2 = torch.tensor(zero_test_set)
Y2 = torch.tensor(test_label)


X2 = X.view(-1, 19*19)
print(X2.shape)
print(X.shape)

# %%
#테스트
from collections import Counter


X2 = X2.to(device)
#%%
print(num_test_data)
print(num_train_data)
with torch.no_grad():
    hypothesis = model(X2.float())
    predicted = np.round(hypothesis.cpu()).float()
    print(len(predicted))
    predict_list=[]
# 각 경기당 10개의 예측결과 개수를 세어서 제일 많은 개수의 급수로 결과값
    for i in range(num_train_data):
        setlist = []
        for j in range(0,10):
            if(predicted[i*10+ j].item() >= 27.):
                setlist.append(26.)
            else:
                setlist.append(predicted[i *10 + j].item())
        cnt = Counter(setlist)
        predict_list.append(cnt.most_common(1)[0][0])
    
    print(len(predict_list))
    #print('Accuracy on test data: ' + str(sum(preddicted==test_label)/len(test_label)))

# %%
