import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from MM_pretrain.util import *
# from resnet_model import resnet34
import math


# def my_relu(data):
#     data[data < 1e-8] = 1e-8
#     return data


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MM_pretrain(torch.nn.Module):
    def __init__(self, params, myvlbert):
        super(MM_pretrain, self).__init__()
        self.params = params
        self.mybert = myvlbert
        self.pair_preject = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, sentence, img_obj, sentence_lens, mask, mode="train"):
        # Get the text features
        # u = self.text_encoder(sentence, sentence_lens, chars)
        # batch_size = sentence.shape[0]
        # if mode == "test":
        #     batch_size = 1
        all_encoder_layers, attention_probs = self.mybert(img_obj, sentence)
        # print(all_encoder_layers.shape)
        pair_out = self.pair_preject(all_encoder_layers[:, :1, :])
        # pair_out = F.relu(pair_out)
        # print(pair_out.shape)
        return pair_out

