from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.data import Token as FlairToken
from flair.data import Sentence as FlairSentence
from torchcrf import CRF
from data.dataset import MyDataPoint, MyPair
import constants


# constants for model
CLS_POS = 0
SUBTOKEN_PREFIX = '##'
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.feat is None:
            return False
    return True


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x


class MyModel(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            encoder_t: PreTrainedModel,
            hid_dim_t: int,
            encoder_v: nn.Module = None,
            hid_dim_v: int = None,
            token_embedding: TokenEmbeddings = None,
            rnn: bool = None,
            crf: bool = None,
            gate: bool = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = token_embedding
        self.proj = nn.Linear(hid_dim_v, hid_dim_t) if encoder_v else None
        self.aux_head = nn.Linear(hid_dim_t, 2)
        if self.token_embedding:
            self.hid_dim_t += self.token_embedding.embedding_length
        if rnn:
            hid_dim_rnn = 256
            num_layers = 2
            num_directions = 2
            self.rnn = nn.LSTM(self.hid_dim_t, hid_dim_rnn, num_layers, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hid_dim_rnn * num_directions, constants.LABEL_SET_SIZE)
        else:
            self.rnn = None
            self.head = nn.Linear(self.hid_dim_t, constants.LABEL_SET_SIZE)
        self.crf = CRF(constants.LABEL_SET_SIZE, batch_first=True) if crf else None
        self.gate = gate
        self.to(device)

    @classmethod
    def from_pretrained(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = 'resources/models'

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)
        encoder_t = AutoModel.from_pretrained(encoder_t_path)
        config = AutoConfig.from_pretrained(encoder_t_path)
        hid_dim_t = config.hidden_size

        if args.encoder_v:
            encoder_v = getattr(torchvision.models, args.encoder_v)()
            encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth'))
            hid_dim_v = encoder_v.fc.in_features
        else:
            encoder_v = None
            hid_dim_v = None

        if args.stacked:
            flair.cache_root = 'resources/models'
            flair.device = device
            token_embedding = StackedEmbeddings([
                WordEmbeddings('crawl'),
                WordEmbeddings('twitter'),
                FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')
            ])
        else:
            token_embedding = None

        return cls(
            device=device,
            tokenizer=tokenizer,
            encoder_t=encoder_t,
            hid_dim_t=hid_dim_t,
            encoder_v=encoder_v,
            hid_dim_v=hid_dim_v,
            token_embedding=token_embedding,
            rnn=args.rnn,
            crf=args.crf,
            gate=args.gate,
        )

    def _bert_forward_with_image(self, inputs, pairs, gate_signal=None):
        images = [pair.image for pair in pairs]
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)
        if not use_cache(self.encoder_v, images):
            visual_embeds = resnet_encode(self.encoder_v, visual_embeds)
        visual_embeds = self.proj(visual_embeds)
        if gate_signal is not None:
            visual_embeds *= gate_signal
        inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)

        batch_size = visual_embeds.size()[0]
        visual_length = visual_embeds.size()[1]

        attention_mask = inputs.attention_mask
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = inputs.token_type_ids
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

    def ner_encode(self, pairs: List[MyPair], gate_signal=None):
        sentence_batch = [pair.sentence for pair in pairs]
        tokens_batch = [[token.text for token in sentence] for sentence in sentence_batch]
        inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt',
                                return_special_tokens_mask=True, return_offsets_mapping=True).to(self.device)

        if self.encoder_v:
            outputs = self._bert_forward_with_image(inputs, pairs, gate_signal)
            feat_batch = outputs.last_hidden_state[:, :-VISUAL_LENGTH]
        else:
            outputs = self.encoder_t(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_type_ids=inputs.token_type_ids,
                return_dict=True
            )
            feat_batch = outputs.last_hidden_state

        ids_batch = inputs.input_ids
        offset_batch = inputs.offset_mapping
        mask_batch = inputs.special_tokens_mask.bool().bitwise_not()
        for sentence, ids, offset, mask, feat in zip(sentence_batch, ids_batch, offset_batch, mask_batch, feat_batch):
            ids = ids[mask]
            offset = offset[mask]
            feat = feat[mask]
            subtokens = self.tokenizer.convert_ids_to_tokens(ids)
            length = len(subtokens)

            token_list = []
            feat_list = []
            i = 0
            while i < length:
                j = i + 1
                # the 'or' condition is for processing Korea characters
                while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)):
                    j += 1
                token_list.append(''.join(subtokens[i:j]))
                feat_list.append(torch.mean(feat[i:j], dim=0))
                i = j
            assert len(sentence) == len(token_list)

            for token, token_feat in zip(sentence, feat_list):
                token.feat = token_feat

            if self.token_embedding is not None:
                flair_sentence = FlairSentence(str(sentence))
                flair_sentence.tokens = [FlairToken(token.text) for token in sentence]
                self.token_embedding.embed(flair_sentence)
                for token, flair_token in zip(sentence, flair_sentence):
                    token.feat = torch.cat((token.feat, flair_token.embedding))

    def ner_forward(self, pairs: List[MyPair]):
        if self.gate:
            tokens_batch = [[token.text for token in pair.sentence] for pair in pairs]
            inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt')
            inputs = inputs.to(self.device)
            outputs = self._bert_forward_with_image(inputs, pairs)
            feats = outputs.last_hidden_state[:, CLS_POS]
            logits = self.aux_head(feats)
            gate_signal = F.softmax(logits, dim=1)[:, 1].view(len(pairs), 1, 1)
        else:
            gate_signal = None

        self.ner_encode(pairs, gate_signal)

        sentences = [pair.sentence for pair in pairs]
        batch_size = len(sentences)
        lengths = [len(sentence) for sentence in sentences]
        max_length = max(lengths)

        feat_list = []
        zero_tensor = torch.zeros(max_length * self.hid_dim_t, device=self.device)
        for sentence in sentences:
            feat_list += [token.feat for token in sentence]
            num_padding = max_length - len(sentence)
            if num_padding > 0:
                padding = zero_tensor[:self.hid_dim_t * num_padding]
                feat_list.append(padding)
        feats = torch.cat(feat_list).view(batch_size, max_length, self.hid_dim_t)

        if self.rnn is not None:
            feats = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
            feats, _ = self.rnn(feats)
            feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)

        logits_batch = self.head(feats)

        labels_batch = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        for i, sentence in enumerate(sentences):
            labels = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            labels_batch[i, :lengths[i]] = labels

        if self.crf:
            mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1
            loss = -self.crf(logits_batch, labels_batch, mask, reduction='mean')
            pred_ids = self.crf.decode(logits_batch, mask)
            pred = [[constants.ID_TO_LABEL[i] for i in ids] for ids in pred_ids]
        else:
            loss = torch.zeros(1, device=self.device)
            for logits, labels, length in zip(logits_batch, labels_batch, lengths):
                loss += F.cross_entropy(logits[:length], labels[:length], reduction='sum')
            loss /= batch_size
            pred_ids = torch.argmax(logits_batch, dim=2).tolist()
            pred = [[constants.ID_TO_LABEL[i] for i in ids[:length]] for ids, length in zip(pred_ids, lengths)]

        return loss, pred

    def itr_forward(self, pairs: List[MyPair]):
        text_batch = [pair.sentence.text for pair in pairs]
        inputs = self.tokenizer(text_batch, padding=True, return_tensors='pt').to(self.device)
        outputs = self._bert_forward_with_image(inputs, pairs)
        feats = outputs.last_hidden_state[:, CLS_POS]
        logits = self.aux_head(feats)

        labels = torch.tensor([pair.label for pair in pairs], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        pred = torch.argmax(logits, dim=1).tolist()

        return loss, pred
