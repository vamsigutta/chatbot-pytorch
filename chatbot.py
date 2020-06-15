import json
import string
import random

import torch

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

file = 'model.pth'
data = torch.load(file)

model = NeuralNetwork(data['input_size'], data['hidden_size'], data['output_size'])
model.load_state_dict(data['model_state'])
all_words = data['all_words']
tags = data['tags']
bot_name = intents['name']

print(f"Hello I am {bot_name}, Please type quit to exit")
while True:
    input_message = input('me :: ')

    if input_message == 'quit':
        break
    # print(f"me :: {input_message}")
    tokenized_sentence = tokenize(input_message)
    tokenized_sentence = bag_of_words(tokenized_sentence, all_words)
    x = tokenized_sentence.reshape(1, tokenized_sentence.shape[0])
    x = torch.from_numpy(x)
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name} :: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name} :: I do not understand....")

