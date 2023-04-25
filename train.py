import numpy as np
import random
import json

# for web scraping
import requests
from bs4 import BeautifulSoup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# web scraping
URL = 'https://www.goodreads.com/quotes/tag/{}?page={}'
page = requests.get(URL.format('inspirational', 1))
soup = BeautifulSoup(page.content, 'html.parser')
quotes = soup.find_all('div', attrs={'class': 'quoteText'})
print(quotes[0].text.strip().split('\n')[0][1:-1])

with open('intents.json', 'r') as f:
    intents = json.load(f)

# save the web scraped quotes
with open('quotes.txt', 'w') as f:
    for quote in quotes:
        f.write(quote.text.strip().split('\n')[0][1:-1] + '\n')


# save the web scraped quotes to intents.json as "tag": "quotes" and the user asking for quotes as "patterns" and the bot's response as "responses"
with open('intents.json', 'r') as f:
    intents = json.load(f)

with open('quotes.txt', 'r') as f:
    quotes = f.readlines()

with open('intents.json', 'w') as f:
    json.dump(intents, f)

for quote in quotes:
    intents['intents'].append({
        "tag": "quotes",
        "patterns": ["I want a quote", "I want a quote on {}".format(quote)],
        "responses": ["Here's a quote: {}".format(quote)]
    })

print('intents.json updated with quotes')

# train the model to get questions from quora and answers and the replies from the comments

# get the questions from quora and the most upvoted answers
URL = 'https://www.quora.com/search?q={}'
page = requests.get(URL.format('how to be happy'))
soup = BeautifulSoup(page.content, 'html.parser')
# get the questions
questions = soup.find_all('span', attrs={'class': 'ui_qtext_rendered_qtext'})
# get the most upvoted answers (15+ upvotes)
upvotes = soup.find_all('span', attrs={'class': 'ui_button_count_inner'})
# get the replies from the comments
answers = soup.find_all('div', attrs={'class': 'ui_qtext_expanded'})
print(questions[0].text.strip())
print(upvotes[0].text.strip())
print(answers[0].text.strip())



all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')