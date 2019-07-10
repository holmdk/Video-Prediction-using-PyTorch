import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, nf, input_size=(64, 64)):
        super(ConvLSTM, self).__init__()
        self.convlstm_1 = ConvLSTMCell(input_size=input_size,
                                       input_dim=1,
                                       hidden_dim=nf,
                                       kernel_size=(3, 3),
                                       bias=True) \
            .cuda()

        self.convlstm_2 = ConvLSTMCell(input_size=input_size,
                                       input_dim=nf,
                                       hidden_dim=nf,
                                       kernel_size=(3, 3),
                                       bias=True) \
            .cuda()

        self.conv2d = nn.Conv2d(in_channels=nf,
                                out_channels=1,
                                bias=False,
                                kernel_size=(3, 3),
                                padding=(1, 1)) \
            .cuda()

        self.conv3d = nn.Conv3d(in_channels=nf,
                                out_channels=1,
                                bias=False,
                                kernel_size=(3, 3, 3),
                                padding=(1, 1, 1)) \
            .cuda()

        self.linear = nn.Linear(nf, 1)

    def forward(self, x, future=0):
        # Inspiration from https://github.com/pytorch/examples/tree/master/time_sequence_prediction

        # initialize hidden state
        h_t, c_t = self.convlstm_1.init_hidden(x.size(0)) # maybe do not initialize to zero but do glorot
        h_t2, c_t2 = self.convlstm_2.init_hidden(x.size(0))

        outputs = []

        seq_len = x.size(1)

        for t in range(seq_len):
            h_t, c_t = self.convlstm_1(input_tensor=x[:, t, :, :, :],
                                       cur_state=[h_t, c_t])

            h_t2, c_t2 = self.convlstm_2(input_tensor=h_t,
                                         cur_state=[h_t2, c_t2])
            output = self.conv2d(h_t2)
            output = nn.Sigmoid()(output) # shouldn't this be sigmoid??
            outputs += [output]

        for i in range(future):  # if we should predict the future
            h_t, c_t = self.convlstm_1(input_tensor=output,
                                       cur_state=[h_t, c_t])
            h_t2, c_t2 = self.convlstm_2(input_tensor=h_t,
                                         cur_state=[h_t2, c_t2])
            output = self.conv2d(h_t2)
            output = nn.Sigmoid()(output) # shouldn't this be sigmoid??
            outputs += [output]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

