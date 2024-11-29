print("::::::::::::::::::::::::: model.py is running ::::::::::::::::::::::::::::::::")
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,dropout_rate=0.1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size*3)
        self.l2 = nn.Linear(hidden_size*3, hidden_size*3)
        self.l3 = nn.Linear(hidden_size*3, hidden_size*3)
        self.l4 = nn.Linear(hidden_size*3, hidden_size*3)
        self.l5 = nn.Linear(hidden_size*3, num_classes)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # print(f"Input to l1: {x.shape}")
        out = self.l1(x)
        # print(f"Output of l1: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after activation

        out = self.l2(out)
        # print(f"Output of l1: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout again if needed

        out = self.l3(out)
        # print(f"Output of l1: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout again if needed

        out = self.l4(out)
        # print(f"Output of l1: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout again if needed

        out = self.l5(out)
        # print(f"Output of l1: {out.shape}")
        return out        