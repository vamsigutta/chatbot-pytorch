import json
import numpy as np
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

## Custom scripts
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

tags = []
all_words = []
xy = []
for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend([word.lower() for word in w])
        xy.append((w,intent['tag']))

ignore_words = string.punctuation
all_words = [stem(word) for word in all_words if word not in ignore_words]
tags = sorted(set(tags))
all_words = sorted(set(all_words))


x_train = []
y_train = []
for (sentences, tag) in xy:
    word_num = bag_of_words(sentences, all_words)
    label = tags.index(tag)
    x_train.append(word_num)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# print(x_train.dtype)
### HyperParameters
batch_size = 8
lr = 0.001
hidden_size = 32
epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        # super().__init__()
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = NeuralNetwork(len(all_words), hidden_size, len(tags))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):
    for (words, labels) in train_loader:
        output = model(words)
        labels = labels.type(torch.int64)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"{epoch + 1} epoch:: Loss {loss.item():.7f}")


data_save = {
    "model_state" : model.state_dict(),
    "input_size"  : len(all_words),
    "hidden_size" : hidden_size,
    "output_size" : len(tags),
    "all_words"   : all_words,
    "tags"        : tags
}

file = "model.pth"
torch.save(data_save, file)
print(f"Weights saved in {file}")