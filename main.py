import os
import argparse
from model import *
from MM_pretrain.model import *
from data_loader import DataLoader
from data_loader_bb import DataLoader as DLbb
from evaluator import Evaluator
from trainer import Trainer
import random
import numpy as np
import torch
import flair

from cfgs.config import config, update_config
from MM_pretrain.resnet_vlbert import ResNetVLBERT


device_id = 2
torch.cuda.set_device(device_id)
flair.device = torch.device('cuda:%d' % device_id)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for MNER')

    parser.add_argument("--pre_image_features_dir", dest="pre_image_features_dir", type=str,
                        default='/home/data/syx/twitter/twitter100k/img_feature/')
    parser.add_argument("--pre_image_obj_features_dir", dest="pre_image_obj_features_dir", type=str,
                        default='/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/wfs/data/twitter100k/twitter100k_image/')
    # parser.add_argument("--image_features_dir", dest="image_features_dir", type=str,
    #                     default='/mnt/wfs/data/twitter10k/npyimg/')
    parser.add_argument("--pre_split_file", dest="pre_split_file", type=str, default='/home/data/syx/twitter/twitter100k/')

    # parser.add_argument("--image_features_dir", dest="image_features_dir", type=str,
    #                     default='/home/data/syx/twitter/aaai/img_obj/')
    # parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
    #                     default='/home/data/syx/twitter/aaai/img_obj/')
    # parser.add_argument("--caption_file", dest="caption_file", type=str, default='///')
    # parser.add_argument("--split_file", dest="split_file", type=str, default='/mnt/wfs/dataz'z'z/mul/unk/')
    # parser.add_argument("--split_file", dest="split_file", type=str,
    #                         default='/home/data/syx/twitter/aaai/')
    # parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
    #                         default='/home/data/syx/twitter/aaai/ner_img/')
    # parser.add_argument("--image_obj_boxes_dir", dest="image_obj_boxes_dir", type=str,
    #                         default="/home/data/syx/twitter/aaai/boxes/")
    parser.add_argument("--split_file", dest="split_file", type=str,
                        default='/home/data/datasets/snap/')
    parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
                        default='/home/data/datasets/snap/ner_img/')
    parser.add_argument("--image_obj_boxes_dir", dest="image_obj_boxes_dir", type=str,
                        default="/home/data/datasets/snap/boxes/")
    # parser.add_argument("--word2vec_model", dest="word2vec_model", type=str, default='/mnt/wfs/data/glove100.txt')
    parser.add_argument("--pre_word2vec_model", dest="pre_word2vec_model", type=str,
                        default='/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/wfs/data/glove100.txt')
    parser.add_argument("--word2vec_model", dest="word2vec_model", type=str, default='/home/iot538/Documents/syx/cr/40Wtweet_200dim.model')

    #parameters for pretrain model
    parser.add_argument("--pretrain_load", dest="pretrain_load", type=int, default=1)
    parser.add_argument("--pre_embedding_dimension_char", dest="pre_embedding_dimension_char", type=int, default=25)
    parser.add_argument("--pre_use_char_cnn", dest="pre_use_char_cnn", type=int, default=0)
    parser.add_argument("--pre_hidden_dimension", dest="pre_hidden_dimension", type=int, default=256)
    parser.add_argument("--pre_embedding_dimension", dest="pre_embedding_dimension", type=int, default=100)
    parser.add_argument("--cat_h_e", dest="cat_h_e", type=int, default=1)
    parser.add_argument("--pretrain_vocab_size", dest="pretrain_vocab_size", type=int, default=0)
    parser.add_argument("--pretrain_model", dest="pretrain_model", type=str,
                        default='/home/data/wjq_new/pretrain_model/VLRB_0.5/model_pre_100k_hidden_256/epoch3_f1_0.86284.pth')
    MODEL_DIR = '/home/data/wjq_new/ner/models_addpre/'

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=512)
    # parser.add_argument("--hidden_dimension_char", dest="hidden_dimension_char", type=int, default=30)
    #for weighted_sum --hidden_dimension_char
    # parser.add_argument("--hidden_dimension_char", dest="hidden_dimension_char", type=int, default=30)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=200)
    parser.add_argument("--embedding_dimension_char", dest="embedding_dimension_char", type=int, default=25)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=30000)
    parser.add_argument("--char_vocab_size", dest="char_vocab_size", type=int, default=127)
    parser.add_argument("--use_char_embedding", dest="use_char_embedding", type=int, default=1)
    # parser.add_argument("--use_filter_gate", dest="use_filter_gate", type=int, default=1)
    # parser.add_argument("--use_only_text", dest="use_only_text", type=int, default=1)
    parser.add_argument("--use_img", dest="use_img", type=int, default=0)
    parser.add_argument("--pre_use_img", dest="pre_use_img", type=int, default=1)
    parser.add_argument("--use_char_cnn", dest="use_char_cnn", type=int, default=0)

    parser.add_argument("--att_head", dest="atthead", type=float, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--lr", dest="lr", type=float, default=5e-5)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=40)
    parser.add_argument("--lambda_1", dest="lambda_1", type=int, default=9)
    parser.add_argument("--pre_n_layers", dest="pre_n_layers", type=int, default=2)
    parser.add_argument("--n_layers", dest="n_layers", type=int, default=3)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=5)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.0000001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=15)
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.01)
    parser.add_argument("--validate_every", dest="validate_every", type=int, default=1)
    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="model_weights.t7")
    parser.add_argument("--sent_maxlen", dest="sent_maxlen", type=int, default=35)
    parser.add_argument("--word_maxlen", dest="word_maxlen", type=int, default=41)
    parser.add_argument("--visual_feature_dimension", dest="visual_feature_dimension", type=int,
                        default=512)
    parser.add_argument("--regions_in_image", dest="regions_in_image", type=int, default=49)

    parser.add_argument('--cfg', type=str, help='path to config file',
                        default='cfgs/base_gt_boxes_4x16G.yaml')
    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    return args, config


def main():
    params, config = parse_arguments()
    print(config)
    print(params)
    print("Constructing data loaders...")

    pre_model = None
    if params.pretrain_load == 1:

        myvlbert = ResNetVLBERT(config)
        pretrained_bert_model = torch.load('/home/data/datasets/embeddings/bert/bert-base-uncased/pytorch_model.bin')
        pretrained_vlbert_model = torch.load('/home/data/wjq/vlbert/model/pretrained_model/vl-bert-base-e2e.model')[
            'state_dict']
        new_state_dict = myvlbert.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            print(k)
            key = k.replace('vlbert', 'bert') \
                .replace('LayerNorm.weight', 'LayerNorm.gamma') \
                .replace('LayerNorm.bias', 'LayerNorm.beta') \
                # .replace('word_embeddings', 'embeddings.word_embeddings')\
            # .replace('position_embeddings', 'position_embeddings.weight')\
            # .replace('embedding_LayerNorm', 'embeddings.LayerNorm')
            keyvl = 'module.' + k
            if key in pretrained_bert_model.keys():
                new_state_dict[k] = pretrained_bert_model[key]
            elif keyvl in pretrained_vlbert_model.keys():
                new_state_dict[k] = pretrained_vlbert_model[keyvl]
            else:
                miss_keys.append(k)
        if len(miss_keys) > 0:
            print('miss keys: {}'.format(miss_keys))
        myvlbert.load_state_dict(new_state_dict)

        pre_model = MM_pretrain(params, myvlbert)
        # pre_model.load_state_dict(pretrain['model_state_dict'])
        print('Load pretrain model...[OK]')

    dl = DataLoader(params)
    dlbb = DLbb(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, config, dl, dlbb, evaluator, pre_model)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        model = MNER(params)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        acc, f1, prec, rec = evaluator.get_accuracy(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    setup_seed(1024)
    main()
