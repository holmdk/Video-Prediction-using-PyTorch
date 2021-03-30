#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2020/03/09 18:47:50
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   encoder
'''

from torch import nn
from model_utils import make_layers
import torch
import logging


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output


#
#
# if __name__ == "__main__":
#     from net_params import convgru_encoder_params, convgru_decoder_params
#     from data.mm import MovingMNIST
#
#     encoder = Encoder(convgru_encoder_params[0],
#                       convgru_encoder_params[1]).cuda()
#     trainFolder = MovingMNIST(is_train=True,
#                               root='data/',
#                               n_frames_input=10,
#                               n_frames_output=10,
#                               num_objects=[3])
#     trainLoader = torch.utils.data.DataLoader(
#         trainFolder,
#         batch_size=4,
#         shuffle=False,
#     )
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
#         inputs = inputVar.to(device)  # B,S,1,64,64
#         state = encoder(inputs)
#


# if __name__ == "__main__":
#     from net_params import convlstm_encoder_params, convlstm_forecaster_params
#     from data.mm import MovingMNIST
#     from encoder import Encoder
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     encoder = Encoder(convlstm_encoder_params[0],
#                       convlstm_encoder_params[1]).cuda()
#     decoder = Decoder(convlstm_forecaster_params[0],
#                       convlstm_forecaster_params[1]).cuda()
#     if torch.cuda.device_count() > 1:
#         encoder = nn.DataParallel(encoder)
#         decoder = nn.DataParallel(decoder)
#
#     trainFolder = MovingMNIST(is_train=True,
#                               root='data/',
#                               n_frames_input=10,
#                               n_frames_output=10,
#                               num_objects=[3])
#     trainLoader = torch.utils.data.DataLoader(
#         trainFolder,
#         batch_size=8,
#         shuffle=False,
#     )
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
#         inputs = inputVar.to(device)  # B,S,1,64,64
#         state = encoder(inputs)
#         break
#     output = decoder(state)
#     print(output.shape)  # S,B,1,64,64
#
#
# class Autoencoder(nn.Module):
#
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, input):
#         state = self.encoder(input)
#         output = self.decoder(state)
#         return output