import torch
from torch import nn


class OnlineLSTM(nn.Module):

    def __init__(self, input_size=4096, hidden_size=128, window_size=60, stride_size=30, pooling="last", metric_size=32):

        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.window_size = window_size
        self.stride_size = stride_size
        self.pooling = pooling
        self.metric_size = metric_size

        self.model = nn.LSTMCell(self.input_size, self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, self.metric_size)

    def output(self, h):

        # return h
        return self.fc(h)

    def forward(self, inp, h1, c1):

        h_last, c_last = None, None

        window_size = min(self.window_size, inp.shape[0])

        out = 0

        for i in range(window_size):

            h1, c1 = self.model(inp[i, :].unsqueeze(0), (h1, c1))

            if i == self.stride_size - 1:
                h_last, c_last = h1.clone().detach(), c1.clone().detach()

            if self.pooling == "mean" and i >= window_size - self.stride_size:
                out += self.output(h1)

        out /= self.stride_size

        if self.pooling == "last":
            out = self.output(h1)

        out = torch.tanh(out)

        return out, h_last, c_last
