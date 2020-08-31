from util import *
from tqdm import tqdm
from torchcrf import CRF
import torch.nn.functional as F


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_accuracy(self, model, split, crf=None):
        if split == 'val':
            data_loader = self.data_loader.val_data_loader
        else:
            data_loader = self.data_loader.test_data_loader

        if crf == None:
            num_of_tags = len(self.data_loader.labelVoc)
            crf = CRF(num_of_tags)
            if torch.cuda.is_available():
                crf = crf.cuda()

        model.eval()
        labels_pred = []
        labels = []
        words = []
        sent_lens = []
        for (x, x_flair, x_obj, y, mask, lens) in tqdm(data_loader):
            pair_out = model(to_variable(x), x_flair, to_variable(x_obj), lens,
                             to_variable(mask), relation=None, mode="test")  # seq_len * bs * labels
            relation = F.softmax(pair_out, dim=-1)
            relation_ = relation.detach()
            emissions, attention_probs = model(to_variable(x), x_flair, to_variable(x_obj), lens, to_variable(mask),
                                               relation=relation_, mode="test")  # seq_len * bs * labels
            pre_test_label_index = crf.decode(emissions)  # bs * seq_len
            words.append(x)
            # words.append(x.cpu().numpy())
            labels.append(y.cpu().numpy().squeeze(0))
            # labels.append(y.cpu().numpy())
            labels_pred.append(pre_test_label_index[0])
            # labels_pred.append(pre_test_label_index)
            sent_lens.append(lens.cpu().numpy()[0])
            # sent_lens.append(lens.cpu().numpy())

        return self.evaluate(labels_pred, labels, words, sent_lens)

    def evaluate(self, labels_pred, labels, words, sents_length):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for lab, lab_pred, length, word_sent in zip(labels, labels_pred, sents_length, words):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            lab_chunks = set(self.get_chunks(lab, self.data_loader.labelVoc))
            lab_pred_chunks = set(self.get_chunks(lab_pred, self.data_loader.labelVoc))
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1, p, r

    def get_chunks(self, seq, tags):
        """
        tags:dic{'per':1,....}
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]
        """
        default = tags['O']
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def get_chunk_type(self, tok, idx_to_tag):
        """
        Args:
            tok: id of token, such as 4
            idx_to_tag: dictionary {4: "B-PER", ...}
        Returns:
            tuple: "B", "PER"
        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type
