import traceback
from numbers import Number
from pytorch_pretrained_bert import BertTokenizer
from util import *
import torch.utils.data
import os
from collections import Counter
import numpy as np
import gensim
from gensim.models import word2vec
from gensim.models import fasttext
import  torch
import random
from flair.data import Sentence
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, params, x, x_flair, y, img_id, s_idx, e_idx):
        self.params = params
        self.x = x
        self.x_flair = x_flair
        # self.obj_x = obj_x
        # self.mask_object = mask_object
        self.y = y
        self.img_id  = img_id
        self.s_idx = s_idx
        self.e_idx = e_idx
        self.num_of_samples = e_idx - s_idx

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        x = self.x[self.s_idx + idx]
        x_flair = self.x_flair[self.s_idx + idx]
        y = self.y[self.s_idx + idx]
        img_id = self.img_id[self.s_idx + idx]
        path = os.path.join(self.params.image_obj_features_dir, img_id + '.jpg')
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # print(i)
        image = transform(image)
        # image = torch.unsqueeze(image, 0)
        obj_x = np.array(image)

        return x, x_flair, y, obj_x

    def collate(self, batch):
        x = np.array([x[0] for x in batch])
        x_flair = [x[1] for x in batch]
        y = np.array([x[2] for x in batch])
        obj_x = np.array([x[3] for x in batch])
        # trunc_boxes = np.zeros((len(batch), max_boxes_len, 4), np.float)
        # trunc_boxes[:, :, :] = -1.5
        # for id, array in enumerate(boxes):
        #     trunc_boxes[id, 0:len(array), :] = array
        # mask_object = np.array([x[7] for x in batch])
        # img_id = np.array([x[6] for x in batch])
        bool_mask = y == 0
        mask = 1 - bool_mask.astype(np.int)

        # index of first 0 in each row, if no zero then idx = -1
        zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(np.int)
        # print(zero_indices)
        input_len = np.zeros(len(batch))
        for i in range(len(batch)):
            if zero_indices[i] == -1:
                input_len[i] = len(x[i])
            else:
                input_len[i] = zero_indices[i]
        sorted_input_arg = np.argsort(-input_len)
        # print(input_len)
        # print(sorted_input_arg)
        # print(np.argsort(input_len))
        # print(sorted_input_arg)
        # Sort everything according to the sequence length
        # x = x[sorted_input_arg]
        # pre_x = pre_x[sorted_input_arg]
        # pre_x_mask = pre_x_mask[sorted_input_arg]
        # print(x)
        # x = sorted(x, key=lambda i:len(i), reverse=True)
        # print(x)
        x = x[sorted_input_arg]
        x_flair = sorted(x_flair, key=lambda i: len(i), reverse=True)
        # print(pre_x)

        # pre_x = sorted(pre_x, key=lambda i:len(i), reverse=True)
        # pre_x_mask = sorted(pre_x_mask, key=lambda i:len(i), reverse=True)
        # print('\n')
        # print(y)
        # print(sorted_input_arg)
        y = y[sorted_input_arg]
        # print(y)
        obj_x = obj_x[sorted_input_arg]
        mask = mask[sorted_input_arg]
        # mask_object = mask_object[sorted_input_arg]
        input_len = input_len[sorted_input_arg]
        # img_id = img_id[sorted_input_arg]

        max_seq_len = int(input_len[0])


        # trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_x_flair = []
        # trunc_pre_x = []
        # trunc_pre_x_mask = []
        trunc_y = np.zeros((len(batch), max_seq_len))
        trunc_mask = np.zeros((len(batch), max_seq_len))
        # print(len(batch))
        for i in range(len(batch)):
            # print('max_seq_len:', max_seq_len)
            # print('x_len:', len(x[0]))
            # print('y:', y)
            trunc_x_flair.append(x_flair[i])
            trunc_x[i] = x[i, :max_seq_len]

            trunc_y[i] = y[i, :max_seq_len]
            trunc_mask[i] = mask[i, :max_seq_len]
        return to_tensor(trunc_x).long(), trunc_x_flair, to_tensor(obj_x), to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
               to_tensor(input_len).int()


class DataLoader:
    def __init__(self, params):
        '''
        self.x : sentence encoding with padding at word level
        self.x_c : sentence encoding with padding at character level
        self.x_img : image features corresponding to the sentences
        self.y : label corresponding to the words in the sentences
        :param params:
        '''
        self.params = params

        self.sentences, self.datasplit, \
            self.x, self.x_flair, self.y, \
            self.num_sentence, self.labelVoc, self.img_id\
            = self.load_data()

        kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}

        dataset_train = CustomDataSet(params, self.x,  self.x_flair, self.y, self.img_id, self.datasplit[0], self.datasplit[1])
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                             batch_size=self.params.batch_size,
                                                             collate_fn=dataset_train.collate,
                                                             shuffle=True, **kwargs)
        dataset_val = CustomDataSet(params, self.x, self.x_flair, self.y,  self.img_id, self.datasplit[1], self.datasplit[2])
        self.val_data_loader = torch.utils.data.DataLoader(dataset_val,
                                                           batch_size=1,
                                                           collate_fn=dataset_val.collate,
                                                           shuffle=False, **kwargs)
        dataset_test = CustomDataSet(params, self.x,  self.x_flair, self.y, self.img_id, self.datasplit[2], self.datasplit[3])
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=1,
                                                            collate_fn=dataset_test.collate,
                                                            shuffle=False, **kwargs)

        # if self.params.word2vec_model != '':
        #     # self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.params.word2vec_model)
        #     model = word2vec.Word2Vec.load(self.params.word2vec_model)


    def load_data(self):
        print('calculating vocabulary...')

        datasplit, sentences, sent_maxlen, word_maxlen, num_sentence,  img_id = self.load_sentence(
            'IMGID', self.params.split_file, 'train', 'dev', 'test')

        labelVoc, labelVoc_inv = self.vocab_bulid(sentences)

        # word_matrix = self.load_word_matrix(vocb, size=self.params.embedding_dimension)

        x, x_flair, y = self.pad_sequence(sentences, labelVoc,
                                               word_maxlen=self.params.word_maxlen, sent_maxlen=sent_maxlen)

        return [sentences, datasplit, x, x_flair, y, num_sentence,
                labelVoc, img_id]

    def load_sentence(self, IMAGEID, tweet_data_dir, train_name, dev_name, test_name):
        """
        read the word from doc, and build sentence. every line contain a word and it's tag
        every sentence is split with a empty line. every sentence begain with an "IMGID:num"

        """
        # IMAGEID='IMGID'
        img_id = []
        sentences = []
        sentence = []
        sent_maxlen = 0
        word_maxlen = 0
        obj_features = []
        # img_feature = []
        datasplit = []
        # mask_object = []

        for fname in (train_name, dev_name, test_name):
            datasplit.append(len(img_id))
            with open(os.path.join(tweet_data_dir, fname), 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.rstrip()
                    if line == '':
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        sentences.append(sentence)
                        sentence = []
                    else:
                        if IMAGEID in line:
                            num = line[6:]
                            img_id.append(num)
                        else:
                            sentence.append(line.split('\t'))
                            word_maxlen = max(word_maxlen, len(str(line.split()[0])))

        # sentences.append(sentence)
        datasplit.append(len(img_id))
        num_sentence = len(sentences)

        print("datasplit", datasplit)
        print(sentences[len(sentences) - 2])
        print(sentences[0])
        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', len(sentences))
        print('number image', len(img_id))
        # mask_object = np.asarray(mask_object)
        # img_id = [int(i) for i in img_id]
        # self.params.word_maxlen = word_maxlen
        # self.params.sent_maxlen = sent_maxlen
        return [datasplit, sentences, sent_maxlen, word_maxlen, num_sentence, img_id]

    def vocab_bulid(self, sentences):
        """
        input:
            sentences list,
            the element of the list is (word, label) pair.
        output:
            some dictionaries.

        """
        words = []
        chars = []
        labels = []

        for sentence in sentences:
            for word_label in sentence:
                words.append(word_label[0])
                labels.append(word_label[1])
                for char in word_label[0]:
                    chars.append(char)

        labels_counts = Counter(labels)
        print('labels_counts', len(labels_counts))
        print(labels_counts)
        labelVoc_inv, labelVoc = self.label_index(labels_counts)
        print('labelVoc', labelVoc)

        return [labelVoc, labelVoc_inv]

    @staticmethod
    def label_index(labels_counts):
        """
           the input is the output of Counter. This function defines the (label, index) pair,
           and it cast our datasets label to the definition (label, index) pair.
        """

        num_labels = len(labels_counts)
        labelVoc_inv = [x[0] for x in labels_counts.most_common()]

        labelVoc = {'0': 0,
                    'B-PER': 1, 'I-PER': 2,
                    'B-LOC': 3, 'I-LOC': 4,
                    'B-ORG': 5, 'I-ORG': 6,
                    'B-OTHER': 7, 'I-OTHER': 8,
                    'O': 9}
        if len(labelVoc) < num_labels:
            for key, value in labels_counts.items():
                if not labelVoc.has_key(key):
                    labelVoc.setdefault(key, len(labelVoc))
        return labelVoc_inv, labelVoc

    @staticmethod
    def pad_sequences(y, sent_maxlen):
        padded = np.zeros((len(y), sent_maxlen))
        for i, each in enumerate(y):
            trunc_len = min(sent_maxlen, len(each))
            padded[i, :trunc_len] = each[:trunc_len]
        return padded.astype(np.int32)

    def pad_sequence(self, sentences, labelVoc, word_maxlen=30,
                     sent_maxlen=35):
        """
            This function is used to pad the word into the same length, the word length is set to 30.
            Moreover, it also pad each sentence into the same length, the length is set to 35.

        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # print(tokenizer.vocab)
        # print(sentences[0])
        x = []
        x_flair = []
        y = []
        for sentence in sentences:
            w_id = []
            y_id = []
            st = Sentence()
            for idx, word_label in enumerate(sentence):
                try:
                    w_id.append(tokenizer.vocab[word_label[0].lower()])
                except Exception as e:
                    w_id.append(tokenizer.vocab['[MASK]'])
                st.add_token(word_label[0])
                y_id.append(labelVoc[word_label[1]])
            # print(w_id)
            # print(sentence)
            x.append(w_id)
            x_flair.append(st)
            y.append(y_id)

        y = self.pad_sequences(y, sent_maxlen)
        x = self.pad_sequences(x, sent_maxlen)

        y = np.asarray(y)
        # mask_object = np.asarray(mask_object)

        return [x, x_flair, y]
