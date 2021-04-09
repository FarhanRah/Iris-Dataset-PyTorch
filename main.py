import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
from sklearn.model_selection import train_test_split


class Model(nn.Module):

    # hl = Hidden Layer
    def __init__(self, input_layer=4, hl1=8, hl2=9, output_layer=3):
        super().__init__()
        self.fc1 = nn.Linear(input_layer, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.out_connection = nn.Linear(hl2, output_layer)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        pred_val = self.out_connection(state)

        return pred_val


# Train the data
model = Model()

data = pandas.read_csv("iris.csv")
X = data.drop('variety', axis=1).values  # Data
y = data['variety'].values  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 100
losses = []

for i in range(EPOCHS):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # We are printing every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i} has loss {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("--------------------------------------")

# Use the test data
with torch.no_grad():
    correct_results = 0

    for i, data in enumerate(X_test):
        y_pred = model.forward(data)
        print(f'{i + 1}) {str(y_pred)}')

        if y_pred.argmax().item() == y_test[i]:
            correct_results += 1

    print(f'We got {correct_results} correct results!')

print("--------------------------------------")

# Now lets use some completely unseen data
with torch.no_grad():
    mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])

    result = model(mystery_iris).argmax().item()
    if result == 0:
        print(f'[INFO] {model(mystery_iris)} is an Iris Setosa.')
    elif result == 1:
        print(f'[INFO] {model(mystery_iris)} is an Iris Virginica.')
    else:
        print(f'[INFO] {model(mystery_iris)} is an Iris Versicolor.')
