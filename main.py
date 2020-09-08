import os
import argparse
from mner import *
from rpbert.bert_rel import *
from data_loader import DataLoader
from data_loader_bb import DataLoader as DLbb
from evaluator import Evaluator
from trainer import Trainer
import random
import numpy as np
import torch
import flair
from cfgs.config import config, update_config
from rpbert.resnet_vlbert import ResNetVLBERT

device_id = 2
torch.cuda.set_device(device_id)
flair.device = torch.device('cuda:%d' % device_id)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for MNER')

    parser.add_argument("--pre_image_obj_features_dir", dest="pre_image_obj_features_dir", type=str,
                        default='datasets/smap/rel_img/')
    parser.add_argument("--pre_split_file", dest="pre_split_file", type=str, default='datasets/smap/')

    # parser.add_argument("--split_file", dest="split_file", type=str,
    #                         default='datasets/fudan/')
    # parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
    #                         default='datasets/fudan/ner_img/')

    parser.add_argument("--split_file", dest="split_file", type=str,
                        default='datasets/snap/')
    parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
                        default='datasets/snap/ner_img/')

    parser.add_argument("--pretrain_load", dest="pretrain_load", type=int, default=1)
    parser.add_argument("--pre_hidden_dimension", dest="pre_hidden_dimension", type=int, default=256)
    parser.add_argument("--cat_h_e", dest="cat_h_e", type=int, default=1)

    MODEL_DIR = '/home/data/wjq_new/ner/models_addpre/'
    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=512)

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--lr", dest="lr", type=float, default=5e-5)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=40)

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
        pretrained_bert_model = torch.load('pretrained/bert-base-uncased/pytorch_model.bin')
        new_state_dict = myvlbert.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            print(k)
            key = k.replace('vlbert', 'bert') \
                .replace('LayerNorm.weight', 'LayerNorm.gamma') \
                .replace('LayerNorm.bias', 'LayerNorm.beta')
            if key in pretrained_bert_model.keys():
                new_state_dict[k] = pretrained_bert_model[key]
            else:
                miss_keys.append(k)
        if len(miss_keys) > 0:
            print('miss keys: {}'.format(miss_keys))
        myvlbert.load_state_dict(new_state_dict)

        pre_model = BertRel(params, myvlbert)
        print('Load pretrain rpbert...[OK]')

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
        print("Loading rpbert...")
        model = MNER(params)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading rpbert...[OK]")

        print("Evaluating rpbert on test set...")
        acc, f1, prec, rec = evaluator.get_accuracy(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("Evaluating rpbert on test set...[OK]")


if __name__ == '__main__':
    main()
