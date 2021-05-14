#%%
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



train_data_path = "./train_data.npy"
train_label_path = "./train_label.npy"

train_data = np.load(train_data_path, allow_pickle=True)
train_label_rawdata = np.load(train_label_path, allow_pickle=True)
num_train_data = len(train_data)

test_data_path = "./test_data.npy"
test_label_path = "./test_label.npy"

test_data = np.load(test_data_path, allow_pickle=True)
test_label_rawdata = np.load(test_label_path, allow_pickle=True)
num_test_data = len(test_data)#원래는 train_data 였음 why?? 

#%%

rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
          '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
          '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
#%%
###
last_train_data = np.zeros( (num_train_data, 10, 19,19) )
last_test_data = np.zeros( (num_test_data, 10, 19,19) )

for i in range(num_train_data):
      for j in range(0,10):
        last_train_data[i, j, :, :] = train_data[i][j]

#np.where(last_train_data == -1 last_train_data)# -1 을 2 로 바꿔줌 
###
#%%
zero_train_set = np.zeros( (num_train_data, 10,19,19) )
for i in range(num_train_data):
    for j in range(9):
        temp_data=train_data[i][j+1]-train_data[i][j]
        zero_train_set[i,j,:,:]=temp_data
    zero_train_set[i,0,:,:]=train_data[i][0]
zero_train_set = np.reshape(zero_train_set,(num_train_data,10*19*19))

zero_train_set.shape
X = torch.tensor(zero_train_set)
X.shape

num_train_data = len(train_data)
train_label = np.zeros(num_train_data )
for i in range(num_train_data):
    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    train_label[i] = train_label_idx

Y = torch.tensor(train_label)
Y.shape
Y=Y.view(-1,1)
#%%
###
for i in range(num_test_data):
  for j in range(0,10):
    last_test_data[i, j, :, :] = test_data[i][j]
###
#np.where(last_test_data == -1, 2, last_test_data)
#%%
###
train_label = np.zeros(num_train_data * 10)
for i in range(num_train_data):
  for j in range(0,10):
    train_label_temp = train_label_rawdata[i]
    train_label_idx = rating.index(train_label_temp)
    train_label[i*10 + j] = train_label_idx
  
test_label = np.zeros(num_test_data * 10)
for i in range(num_test_data):
  for j in range(0,10):
    test_label_temp = test_label_rawdata[i]
    test_label_idx = rating.index(test_label_temp)
    test_label[i * 10 + j] = test_label_idx

test_label2 = np.zeros(num_test_data)
for i in range(num_test_data):
  test_label_temp = test_label_rawdata[i]
  test_label_idx = rating.index(test_label_temp)
  test_label2[i] = test_label_idx
###
#%%
###

Y_test = torch.tensor(test_label2)
Y_test = Y_test.view(-1,1)

X = torch.tensor(last_train_data)
Y = torch.tensor(train_label)

X2 = torch.tensor(last_test_data)
Y2 = torch.tensor(test_label)

X = X.view(-1 , 1,  19 , 19)
Y = Y.view(-1,1)

X2 = X2.view(-1, 1, 19,19)
Y2 = Y2.view(-1,1)

print(len(X))
print(len(Y))

#%%
class CNN(torch.nn.Module):
    
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(5, 5,8, 64)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(5,5, 64, 64)
    self.conv3 = nn.Conv2d(5,5,64,64)
    self.conv4 = nn.Conv2d(5,5,64,48)
    self.conv5 = nn.Conv2d(5,5,48,48)
    self.fc1 = nn.Linear(128, 640)
    self.fc2 = nn.Linear(640, 320)
    self.fc3 = nn.Linear(320, 64)
    self.fc4 = nn.Linear(64, 1)
  
  def forward(self, x):
    x = x.view(-1,19,19,10)
    x = self.conv1(x)
    x = F.elu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = F.elu(x)
    x = self.pool(x)
    x = self.conv3(x)
    x = F.elu(x)
    x = x.view(-1, 128  * 1 * 1)
    x = self.fc1(x)
    x = F.elu(x)
    x = self.fc2(x)
    x = F.elu(x)
    x = self.fc3(x)
    x = torch.sigmoid(x)
    x = self.fc4(x)
    return x

#%%
class CNN(torch.nn.Module):
    
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 2)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.fc1 = nn.Linear(128, 640)
    self.fc2 = nn.Linear(640, 320)
    self.fc3 = nn.Linear(320, 64)
    self.fc4 = nn.Linear(64, 1)
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.elu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = F.elu(x)
    x = self.pool(x)
    x = self.conv3(x)
    x = F.elu(x)
    x = x.view(-1, 128  * 1 * 1)
    x = self.fc1(x)
    x = F.elu(x)
    x = self.fc2(x)
    x = F.elu(x)
    x = self.fc3(x)
    x = torch.sigmoid(x)
    x = self.fc4(x)

    return x





#%%
print(Y.shape)
ds = TensorDataset(X,Y);
data_loader = torch.utils.data.DataLoader(dataset=ds,batch_size=1024,shuffle=True)
#%%
print(len(data_loader))
#%%
num_test_data = len(test_data)
zero_test_set = np.zeros( (num_test_data, 10,19,19) )
for i in range(num_test_data):
    for j in range(9):
        temp_data=test_data[i][j+1]-test_data[i][j]
        zero_test_set[i,j,:,:]=temp_data
    zero_test_set[i,0,:,:]=test_data[i][0]
X2 = torch.tensor(zero_test_set)

#%%
X2 = torch.tensor(last_test_data)
Y2 = torch.tensor(test_label)
X2 = X2.view(-1, 1, 19,19)
Y2 = Y2.view(-1,1)

#%%
device = torch.device('cuda')
model = CNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
model.train()

#%%
for epoch in range(1000):
  
  for x , y in data_loader:
    x= x.to('cuda')
    y= y.to('cuda')
    optimizer.zero_grad()
    hypothesis = model(x.float())
    cost = criterion(hypothesis, y.float())
    optimizer.step()
  print(epoch, cost.item())
  
#%%
#테스트
from collections import Counter

X2 = X2.to(device)

with torch.no_grad():
    hypothesis = model(X2.float())
    #print(hypothesis.shape)
    predicted = np.round(hypothesis.cpu()).float()
    predict_list=[]
# 각 경기당 12개의 예측결과 개수를 세어서 제일 많은 개수의 급수로 결과값
    for i in range(num_test_data):
      setlist = []
      for j in range(0,10):
        setlist.append(predicted[i * 10 + j].item())
      cnt = Counter(setlist)
      predict_list.append(cnt.most_common(1)[0][0])
      #print(predict_list)
    predict_list = torch.FloatTensor(predict_list)
    predict_list = predict_list.view(-1, 1)
    accuracy = sum(predict_list == Y_test)/(num_test_data)
    #print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    #print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    #print('실제값(Y): ', Y2.cpu().numpy())
    #print('정확도(Accuracy): ', accuracy)
    #print('Accuracy on test data: ' + str(sum(predict_list==Y2)/num_test_data))

# %%
    print(accuracy.item())
    #cnt = Counter(predict_list)
    print(sum(predict_list==0.))

# %%
