import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        DROPOUT_P = config["dropout"]

        self.input_size = input_size # from main.py, this is the length of char_set, which is a set of unique characters found in the dataset
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size # same as input_size
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT_P
        
        """
        Complete the code

        TODO: 
        (i) Initialize embedding layer with input_size and hidden_size
        (ii) Initialize the recurrent layer based on model type (i.e., LSTM) using hidden size and num_layers
        (iii) Initialize linear output layer using hidden size and output size
        (iv) Initialize dropout layer with dropout probability
        """
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.hidden = None
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers)
            
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.drop = nn.Dropout(p=self.dropout)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.

        Initialise with zeros.
        """
        # initialize hidden state
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device) # for unbatched input
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
        self.hidden = (h0, c0)
        
        
    def forward(self, seq):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)

        Returns:
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values


        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """
        x0 = self.embedding(seq)
        x1, self.hidden = self.rnn(x0, self.hidden) # x1 has hidden_size neurons
        x2 = self.drop(x1)
        out = self.linear(x2)

        return out, x1.cpu().detach().numpy()