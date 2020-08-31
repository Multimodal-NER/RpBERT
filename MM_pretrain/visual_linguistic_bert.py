import torch
import torch.nn as nn
from external.pytorch_pretrained_bert.modeling import BertLayerNorm, BertEncoder, BertPooler, ACT2FN, BertOnlyMLMHead
import torch.nn.functional as F
# todo: add this to config
NUM_SPECIAL_WORDS = 1000


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        self.config = config
        super(BaseModel, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, *args, **kwargs):
        raise NotImplemented


class VisualLinguisticBert(BaseModel):
    def __init__(self, config):
        super(VisualLinguisticBert, self).__init__(config)

        self.config = config

        # embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.end_embedding = nn.Embedding(1, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # for compatibility of roberta
        self.position_padding_idx = config.position_padding_idx

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.visual_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.visual_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.visual_size, config.hidden_size)
        if config.visual_ln:
            self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            visual_scale_text = nn.Parameter(torch.as_tensor(self.config.visual_scale_text_init, dtype=torch.float),
                                             requires_grad=True)
            self.register_parameter('visual_scale_text', visual_scale_text)
            visual_scale_object = nn.Parameter(torch.as_tensor(self.config.visual_scale_object_init, dtype=torch.float),
                                               requires_grad=True)
            self.register_parameter('visual_scale_object', visual_scale_object)

        self.encoder = BertEncoder(config)

        if self.config.with_pooler:
            self.pooler = BertPooler(config)

        # init weights
        self.apply(self.init_weights)
        if config.visual_ln:
            self.visual_ln_text.weight.data.fill_(self.config.visual_scale_text_init)
            self.visual_ln_object.weight.data.fill_(self.config.visual_scale_object_init)

        if config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False
            self.special_word_embeddings = nn.Embedding(NUM_SPECIAL_WORDS, config.hidden_size)
            self.special_word_embeddings.weight.data.copy_(self.word_embeddings.weight.data[:NUM_SPECIAL_WORDS])

    def word_embeddings_wrapper(self, input_ids):
        if self.config.word_embedding_frozen:
            word_embeddings = self.word_embeddings(input_ids)
            word_embeddings[input_ids < NUM_SPECIAL_WORDS] \
                = self.special_word_embeddings(input_ids[input_ids < NUM_SPECIAL_WORDS])
            return word_embeddings
        else:
            return self.word_embeddings(input_ids)

    def forward(self,
                text_input_ids,
                text_token_type_ids,
                text_visual_embeddings,
                text_mask,
                object_vl_embeddings,
                object_mask,
                output_all_encoded_layers=True,
                output_text_and_object_separately=False,
                output_attention_probs=True,
                relation=None):

        # get seamless concatenate embeddings and mask
        embedding_output, attention_mask, text_mask_new, object_mask_new = self.embedding(text_input_ids,
                                                                                          text_token_type_ids,
                                                                                          text_visual_embeddings,
                                                                                          text_mask,
                                                                                          object_vl_embeddings,
                                                                                          object_mask,
                                                                                          relation)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        embedding_mask = attention_mask.squeeze(1).transpose(1, 2).to(dtype=next(self.parameters()).dtype)
        embedding_output = embedding_output * embedding_mask
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = attention_mask > 0
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_attention_mask = 1.0 - extended_attention_mask
        # extended_attention_mask[extended_attention_mask == 0] = -1e9
        # extended_attention_mask = torch.log(
        #     extended_attention_mask
        # )

        if output_attention_probs:
            encoded_layers, attention_probs = self.encoder(embedding_output,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers=output_all_encoded_layers,
                                                           output_attention_probs=output_attention_probs)
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers,
                                          output_attention_probs=output_attention_probs)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if output_text_and_object_separately:
            if not output_all_encoded_layers:
                encoded_layers = [encoded_layers]
            encoded_layers_text = []
            encoded_layers_object = []
            for encoded_layer in encoded_layers:
                max_text_len = text_input_ids.shape[1]
                max_object_len = object_vl_embeddings.shape[1]
                encoded_layer_text = encoded_layer[:, :max_text_len]
                encoded_layer_object = encoded_layer.new_zeros(
                    (encoded_layer.shape[0], max_object_len, encoded_layer.shape[2]))
                encoded_layer_object[object_mask] = encoded_layer[object_mask_new]
                encoded_layers_text.append(encoded_layer_text)
                encoded_layers_object.append(encoded_layer_object)
            if not output_all_encoded_layers:
                encoded_layers_text = encoded_layers_text[0]
                encoded_layers_object = encoded_layers_object[0]
            if output_attention_probs:
                return encoded_layers_text, encoded_layers_object, pooled_output, attention_probs
            else:
                return encoded_layers_text, encoded_layers_object, pooled_output
        else:
            if output_attention_probs:
                return encoded_layers, pooled_output, attention_probs
            else:
                return encoded_layers, pooled_output

    def embedding(self,
                  text_input_ids,
                  text_token_type_ids,
                  text_visual_embeddings,
                  text_mask,
                  object_vl_embeddings,
                  object_mask,
                  relation=None):

        text_linguistic_embedding = self.word_embeddings_wrapper(text_input_ids)
        if self.visual_1x1_text is not None:
            text_visual_embeddings = self.visual_1x1_text(text_visual_embeddings)
        if self.config.visual_ln:
            text_visual_embeddings = self.visual_ln_text(text_visual_embeddings)
        else:
            text_visual_embeddings *= self.visual_scale_text
        text_vl_embeddings = text_linguistic_embedding + text_visual_embeddings

        object_visual_embeddings = object_vl_embeddings[:, :, :self.config.visual_size]
        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(object_visual_embeddings)
        if self.config.visual_ln:
            object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)
        else:
            object_visual_embeddings *= self.visual_scale_object
        object_linguistic_embeddings = object_vl_embeddings[:, :, self.config.visual_size:]
        object_vl_embeddings = object_linguistic_embeddings + object_visual_embeddings

        bs = text_vl_embeddings.size(0)
        vl_embed_size = text_vl_embeddings.size(-1)
        max_length = (text_mask.sum(1) + object_mask.sum(1)).max()
        grid_ind, grid_pos = torch.meshgrid(torch.arange(bs, dtype=torch.long, device=text_vl_embeddings.device),
                                            torch.arange(max_length, dtype=torch.long, device=text_vl_embeddings.device))
        text_end = text_mask.sum(1, keepdim=True)
        object_end = text_end + object_mask.sum(1, keepdim=True) - 1

        # seamlessly concatenate visual linguistic embeddings of text and object
        _zero_id = torch.zeros((bs, ), dtype=torch.long, device=text_vl_embeddings.device)
        vl_embeddings = text_vl_embeddings.new_zeros((bs, max_length, vl_embed_size))
        vl_embeddings[grid_pos < text_end] = text_vl_embeddings[text_mask]
        vl_embeddings[(grid_pos >= text_end) & (grid_pos <= object_end)] = object_vl_embeddings[object_mask]
        # vl_embeddings[grid_pos == object_end] = self.end_embedding(_zero_id)

        # token type embeddings/ segment embeddings
        token_type_ids = text_token_type_ids.new_zeros((bs, max_length))
        token_type_ids[grid_pos < text_end] = 1
        token_type_ids[(grid_pos >= text_end) & (grid_pos <= object_end)] = 2
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # position embeddings
        position_ids = grid_pos + self.position_padding_idx + 1
        if self.config.obj_pos_id_relative:
            position_ids[(grid_pos >= text_end) & (grid_pos <= object_end)] \
                = text_end.expand((bs, max_length))[(grid_pos >= text_end) & (grid_pos <= object_end)] \
                + self.position_padding_idx + 1
            # position_ids[grid_pos == object_end] = (text_end + 1).squeeze(1) + self.position_padding_idx + 1
        else:
            assert False, "Don't use position id 510/511 for objects and [END]!!!"
            position_ids[(grid_pos >= text_end) & (grid_pos <= object_end)] = self.config.max_position_embeddings - 2
            # position_ids[grid_pos == object_end] = self.config.max_position_embeddings - 1

        position_embeddings = self.position_embeddings(position_ids)
        mask = text_mask.new_zeros((bs, max_length))
        if relation is None:
            mask[grid_pos <= object_end] = 1
        else:
            relation = relation[:, :, :1].view(bs, 1)
            relation = relation.repeat(1, mask.shape[1])
            mask = relation
            mask[grid_pos < text_end] = 1
            mask[grid_pos > object_end] = 0

        embeddings = vl_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings, mask, grid_pos < text_end, (grid_pos >= text_end) & (grid_pos <= object_end)