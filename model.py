# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMModel(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers = 1):
		super(LSTMModel, self).__init__()
		self.num_classes = output_dim #number of classes
		self.num_layers = n_layers #number of layers
		self.input_size = vocab_size #input size
		self.hidden_size = hidden_dim #hidden state
		self.embedding_size = embedding_dim #sequence length

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
							num_layers=n_layers, batch_first=True) #lstm
		self.fc1 =  nn.Linear(hidden_dim, 92) #fully connected 1
		self.fc2 =  nn.Linear(92, output_dim) #fully connected 2

		self.relu = nn.ReLU()

	def forward(self,x):
		h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
		c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
		# Propagate input through LSTM
		out = self.embedding(x)
		output, (hn, cn) = self.lstm(out, (h_0, c_0)) #lstm with input, hidden, and internal state
		out = self.relu(output[:,-1,:])
		# out = self.relu(output[:,-1,:])
		out = self.fc1(out) #first Dense
		out = self.relu(out) #relu
		out = self.fc2(out) #Final Output
		return out

