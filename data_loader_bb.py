from util import *
import torch.utils.data
import os
from collections import Counter
import numpy as np
import gensim
from gensim.models import word2vec
import traceback
# from  MM_pretrain.resnet import trans_image
import random
import torchvision.transforms as transforms
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, params, x, img_id, y, ifparis, s_idx, e_idx):
        self.params = params
        self.x = x
        # self.img_x = img_x
        self.img_id = img_id
        self.y = y
        # self.mask_object = mask_object
        self.s_idx = s_idx
        self.e_idx = e_idx
        self.num_of_samples = e_idx - s_idx
        self.ifpairs = ifparis

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        x = self.x[self.s_idx + idx]
        y = self.y[self.s_idx + idx]
        # img_x = self.img_x[self.s_idx + idx]
        img_id = self.img_id[self.s_idx + idx]
        path = os.path.join(self.params.pre_image_obj_features_dir, img_id+'.jpg')
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # print(i)
        image = transform(image)
        # image = torch.unsqueeze(image, 0)
        obj_x = np.array(image)
        # mask_object = self.mask_object[self.s_idx + idx]
        ifpairs = self.ifpairs[self.s_idx + idx]
        return x, y, obj_x, ifpairs

    def collate(self, batch):
        x = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        # print("xxxxxxx\n", x)
        # print("yyyyyyyyy\n", y)
        # img_x = np.array([x[2] for x in batch])
        obj_x = np.array([x[2] for x in batch])
        # mask_object = np.asarray(x[4] for x in batch)
        # mask_object = np.array([x[5] for x in batch])
        ifpairs = np.array([x[3] for x in batch])
        # bool_mask = x == 0
        bool_mask = y == 0
        mask = 1 - bool_mask.astype(np.int)

        # index of first 0 in each row, if no zero then idx = -1
        zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(np.int)
        input_len = np.zeros(len(batch))
        for i in range(len(batch)):
            if zero_indices[i] == -1:
                # input_len[i] = len(x[i])
                input_len[i] = len(y[i])
            else:
                input_len[i] = zero_indices[i]
        sorted_input_arg = np.argsort(-input_len)

        # Sort everything according to the sequence length
        x = x[sorted_input_arg]
        y = y[sorted_input_arg]
        # img_x = img_x[sorted_input_arg]
        obj_x = obj_x[sorted_input_arg]

        # mask_object = mask_object[sorted_input_arg]
        mask = mask[sorted_input_arg]
        input_len = input_len[sorted_input_arg]
        ifpairs = ifpairs[sorted_input_arg]

        max_seq_len = int(input_len[0])

        trunc_x = np.zeros((len(batch), max_seq_len))
        # trunc_x_mask = np.zeros((len(batch), max_seq_len))
        trunc_y = np.zeros((len(batch), max_seq_len))
        trunc_mask = np.zeros((len(batch), max_seq_len))
        for i in range(len(batch)):
            trunc_x[i] = x[i, :max_seq_len]
            trunc_y[i] = y[i, :max_seq_len]
            trunc_mask[i] = mask[i, :max_seq_len]

        return to_tensor(trunc_x).long(), to_tensor(obj_x), to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
               to_tensor(input_len).int(), to_tensor(ifpairs).long()


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
            self.x, self.img_id, self.y, \
            self.num_sentence, self.ifpairs \
            = self.load_data()
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        ##if for pair_out training, change to self.datasplit[0], self.datasplit[2]
        dataset_train_phase1 = CustomDataSet(params, self.x, self.img_id, self.y, self.ifpairs, self.datasplit[0], self.datasplit[1])
        self.train_phase1_loader = torch.utils.data.DataLoader(dataset_train_phase1,
                                                             batch_size=self.params.batch_size,
                                                             collate_fn=dataset_train_phase1.collate,
                                                             shuffle=True, **kwargs)

        # dataset_train_phase2 = CustomDataSet(params, self.x, self.x_c,self.img_id, self.y,  self.ifpairs, self.datasplit[1], self.datasplit[2])
        # self.train_phase2_loader = torch.utils.data.DataLoader(dataset_train_phase2,
        #                                                        batch_size=self.params.batch_size,
        #                                                        collate_fn=dataset_train_phase2.collate,
        #                                                        shuffle=True, **kwargs)
        # self.val_data_loader = torch.utils.data.DataLoader(dataset_val,
        #                                                    batch_size=1,
        #                                                    collate_fn=dataset_val.collate,
        #                                                    shuffle=False, **kwargs)
        # dataset_test = CustomDataSet(params, self.x, self.x_c, self.img_x, self.y, self.datasplit[2], self.datasplit[3])
        # self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
        #                                                     batch_size=1,
        #                                                     collate_fn=dataset_test.collate,
        #                                                     shuffle=False, **kwargs)

        # if self.params.word2vec_model != '':
        #     self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.params.word2vec_model)
        #     # model = word2vec.Word2Vec.load(self.params.word2vec_model)

    def load_data(self):
        print('calculating vocab  ulary...')
        datasplit, sentences, img_id, sent_maxlen, word_maxlen, num_sentence,  ifpairs = self.load_sentence(
            'IMGID', self.params.pre_split_file, 'twitter100k_relation_image', 'textimage-data-image')
        # print(sent_maxlen)
        # id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = self.vocab_bulid(sentences)
        x, img_id, y = self.pad_sequence(sentences, img_id,
                                             word_maxlen=word_maxlen, sent_maxlen=sent_maxlen)
        return [sentences, datasplit, x,img_id, y, num_sentence,
                ifpairs]

    def load_sentence(self, IMAGEID, tweet_data_dir, train_phase1_name, train_phase2_name):
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
        datasplit = []
        # mask_object = []
        ifpairs =  []
        # img_feature = []
        for fname in (train_phase2_name, ):
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
                            line = line.split("\t")
                            num = line[0][6:]
                            img_id.append(num)
                            ifpairs.append(int(line[1]))
                        else:
                            sentence.append(line.split('\t'))
                            word_maxlen = max(word_maxlen, len(str(line.split()[0])))
            print(1)
            # sentences.append(sentence)
        # datasplit.append(int(len(img_id)/2))
        # datasplit.append(103576)
        datasplit.append(len(img_id))
        num_sentence = len(sentences)

        print("datasplit", datasplit)
        print(sentences[len(sentences) - 2])
        print(sentences[0])

        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', len(sentences))
        print('number image', len(img_id))

        return [datasplit, sentences,  img_id, sent_maxlen, word_maxlen, num_sentence, ifpairs]


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

    def pad_sequence(self, sentences, img_id, word_maxlen=30, sent_maxlen=35):
        """
            This function is used to pad the word into the same length, the word length is set to 30.
            Moreover, it also pad each sentence into the same length, the length is set to 35.

        """

        print(sentences[0])
        x = []
        y = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for sentence in sentences:
            w_id = []
            # w_mask = []
            for word_label in sentence:
                w_id.append(word_label[0])
            w_id = w_id[:-1]

            bert_tokenization = []

            for token in w_id:
                subtokens = tokenizer.tokenize(token)
                bert_tokenization.extend(subtokens)
            if len(bert_tokenization) > sent_maxlen:
                sent_maxlen = len(bert_tokenization)

            tokens = list()
            tokens.append("[CLS]")
            for token in bert_tokenization:
                tokens.append(token)
            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            x.append(input_ids[:])
            y.append(input_ids[:])

        sent_maxlen += 2

        y = self.pad_sequences(y, sent_maxlen)
        x = self.pad_sequences(x, sent_maxlen)


        x = np.asarray(x)
        y = np.asarray(y)
        # mask_object = np.asarray(mask_object)
        return [x, img_id, y]
