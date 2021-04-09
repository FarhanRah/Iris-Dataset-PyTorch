import torch
import torch.nn as nn
import torch.nn.functional as F
# import pandas
# from sklearn.model_selection import train_test_split


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
