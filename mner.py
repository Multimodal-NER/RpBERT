import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Union, Dict, Tuple
# from rpbert.resnet_model import resnet34
import math
from pytorch_pretrained_bert import BertTokenizer
import flair


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BertInputFeatures(object):
    """Private helper class for holding BERT-formatted features"""

    def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
    ):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.token_subtoken_count = token_subtoken_count


class MNER(torch.nn.Module):
    def __init__(self, params,  embeddings, pretrain_model, pretrained_weight=None, num_of_tags=10):
        super(MNER, self).__init__()
        self.params = params

        self.embeddings = embeddings
        self.layer_indexes = [-1]
        self.pooling_operation = "first"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.name = 'bert-base-uncased'

        self.input_embeddding_size = 768 + embeddings.embedding_length
        lstm_input_size = self.input_embeddding_size
        if self.params.pretrain_load == 1:
            self.pretrain_model = pretrain_model
        self.dropout = nn.Dropout(params.dropout)

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=params.hidden_dimension // 2,
                            num_layers=params.n_layers, bidirectional=True)

        self.projection = nn.Linear(in_features=params.hidden_dimension, out_features=num_of_tags)

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0: (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )
        return features

    def _add_embeddings_internal(self, sentences, img_obj, relation=None):
        # sentences = [sentence for sentence in x_flair]
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        if relation is None:
            pair_out = self.pretrain_model(all_input_ids, img_obj, None, None)
            return pair_out

        all_encoder_layers, attention_probs = self.pretrain_model.mybert(img_obj, all_input_ids, relation)

        for sentence_index, sentence in enumerate(sentences):

            feature = features[sentence_index]

            # get aggregated embeddings for each BERT-subtoken in sentence
            subtoken_embeddings = []
            for token_index, _ in enumerate(feature.tokens):
                subtoken_embeddings.append(all_encoder_layers[sentence_index][token_index])

            # get the current sentence object
            token_idx = 0
            for token in sentence:
                # add concatenated embedding to sentence
                token_idx += 1

                if self.pooling_operation == "first":
                    # use first subword embedding if pooling operation is 'first'
                    token.set_embedding(self.name, subtoken_embeddings[token_idx])
                else:
                    # otherwise, do a mean over all subwords in token
                    embeddings = subtoken_embeddings[
                                 token_idx: token_idx
                                            + feature.token_subtoken_count[token.idx]
                                 ]
                    embeddings = [
                        embedding.unsqueeze(0) for embedding in embeddings
                    ]
                    mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                    token.set_embedding(self.name, mean)

                token_idx += feature.token_subtoken_count[token.idx] - 1

        return attention_probs

    def forward(self, sentences, x_flair, img_obj, sentence_lens, mask, relation=None, mode="train"):  # !!! word_seq  char_seq
        if relation is None:
            return self._add_embeddings_internal(x_flair, img_obj, relation)

        self.embeddings.embed(x_flair)
        attention_probs = self._add_embeddings_internal(x_flair, img_obj, relation)

        lengths = [len(sentence.tokens) for sentence in x_flair]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.input_embeddding_size * longest_token_sequence_in_batch,
            dtype=torch.float,
        )

        all_embs = list()
        for sentence in x_flair:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.input_embeddding_size * nb_padding_tokens
                    ].cuda()
                all_embs.append(t)

        embed_flair = torch.cat(all_embs).view(
            [
                len(x_flair),
                longest_token_sequence_in_batch,
                self.input_embeddding_size,
            ]
        )

        embeds = self.dropout(embed_flair)
        embeds = embeds.permute(1, 0, 2)  # se bs hi+embedding_h+c
        packed_input = pack_padded_sequence(embeds, sentence_lens.numpy())
        packed_outputs, _ = self.lstm(packed_input)
        outputs, _ = pad_packed_sequence(packed_outputs)

        outputs = outputs.permute(1, 0, 2)  # batch_size * seq_len * hidden_dimension*2
        outputs = self.dropout(outputs)
        out = self.projection(outputs)

        return out.permute(1, 0, 2), attention_probs        # seq_len * bs * tags
