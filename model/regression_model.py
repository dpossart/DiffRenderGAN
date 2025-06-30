import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=128, dropout_p=0.2):
        """Initialize RegressionModel

            Args:
                n_in: Number of inputs.
                n_out: Number of output parameters.
                dropout_p: Dropout probability.
        """
        super(RegressionModel, self).__init__()
        self.fc1 = nn.utils.weight_norm(nn.Linear(n_in, n_hidden))
        self.fc2 = nn.utils.weight_norm(nn.Linear(n_hidden, n_hidden))
        self.fc3 = nn.utils.weight_norm(nn.Linear(n_hidden, n_hidden))
        self.fc4 = nn.utils.weight_norm(nn.Linear(n_hidden, n_hidden))

        self.output_layer = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(n_hidden, n_out)),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.output_layer(x)