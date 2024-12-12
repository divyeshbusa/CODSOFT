import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize,lemmatize

with open('intents.json','r') as f:
    intents = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
data = torch.load(FILE, weights_only=True)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict (model_state)
model.eval()

bot_name = 'jacky dada'
print("Let's chat! type 'quit' to")
while True:
    sentence = input("you : ")
    if sentence.lower() == 'quit':
        print('bye bye bhidu...')
        break

    sentence = tokenize(sentence)
    sentence = [lemmatize(w) for w in sentence]
    
    X = bag_of_words(sentence, all_words)
    # print(f"x.shape : {X.shape[0]}")
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device, dtype=torch.float) 

    output = model(X)
    # print(f"output : {output}")
    output = output.squeeze(1) 
    # print(f"output.squeeze : {output}")
    _, predicted = torch.max(output, dim=1)
    # print(f"Predicted tensor: {predicted}, Predicted shape: {predicted.shape}") 
    predicted_class = predicted.item()
    # print(f"Predicted class: {predicted_class}")



    tag = tags[predicted_class] 

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted_class]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent['tag']:
                # print(f"predicted: {predicted}")
                # print(f"Patterns: {intent['patterns']}")
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Kya bol raha hai bhidu? samaj nahi aa rahaa..")
