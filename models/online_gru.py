import torch
from torch import nn

from models.online_rnn import OnlineRNN


class OnlineGRU(OnlineRNN):

    def __init__(self, input_size=4096, hidden_size=128, window_size=60, stride_size=30, pooling="last", metric_size=32):

        super().__init__(input_size=input_size, hidden_size=hidden_size, window_size=window_size,
                          stride_size=stride_size, pooling=pooling, metric_size=metric_size)

        self.model = nn.GRUCell(self.input_size, self.hidden_size)
