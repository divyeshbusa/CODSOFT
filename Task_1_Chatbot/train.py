# print("::::::::::::::::::::::::: train.py is running ::::::::::::::::::::::::::::::::")

import json
from nltk_utils import tokenize, lemmatize, bag_of_words
import numpy as np    
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


with open ('intents.json', 'r') as f:
    intents = json.load(f)



all_word = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_word.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '.', '!', ',' , '\'']
all_word = [ lemmatize(w) for w in all_word if w not in ignore_words ]
all_word = sorted(set(all_word))
tags = sorted(set(tags))
# print(all_word,'\n')
# print(tags)

X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence,all_word)
    X_train.append(bag)

    lable = tags.index(tag)
    Y_train.append(lable)

print("X_train :", X_train,"\n")
print("Y_train :", Y_train,"\n")

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("X_train in numpy formet:", X_train,"\n")
print("Y_train in numpy formet:", Y_train,"\n")

class ChatDataset(Dataset):

    def __init__(self):
        
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

# print(input_size, len(all_word))
# print(output_size, tags)


dataset = ChatDataset()
train_loader = DataLoader(dataset , batch_size=64, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate,weight_decay=1e-5)

for epoch in range(num_epochs):
    model.train()
    for (words, lables) in train_loader:
        words = words.to(device)
        lables = lables.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs,lables)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss= {loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size' : input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words' : all_word,
    'tags' : tags    
}


FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved at {FILE}')