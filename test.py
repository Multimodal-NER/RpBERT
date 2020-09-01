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
from flair.embeddings import StackedEmbeddings, WordEmbeddings, CharacterEmbeddings

import subprocess

from cfgs.config import config, update_config

from MM_pretrain.resnet_vlbert import ResNetVLBERT


device_id = 1
torch.cuda.set_device(device_id)
flair.device = torch.device('cuda:%d' % device_id)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for MNER')
    parser.add_argument("--scatter_name", dest="scatter_name", type=str, default="scatter_snap.jpg")
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

    #parameters for pretrain model
    parser.add_argument("--pretrain_load", dest="pretrain_load", type=int, default=1)

    parser.add_argument("--pre_hidden_dimension", dest="pre_hidden_dimension", type=int, default=256)
    parser.add_argument("--pre_embedding_dimension", dest="pre_embedding_dimension", type=int, default=100)
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
    parser.add_argument("--mode", dest="mode", type=int, default=1)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="epoch18_f1_0.87156.pth")
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



    myvlbert = ResNetVLBERT(config)
    pre_model = MM_pretrain(params, myvlbert)

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
        embedding_types = [
            WordEmbeddings(
                '/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/xj/wnut17_advanced/pretrain/en-fasttext-crawl-300d-1M'),
            CharacterEmbeddings('/home/iot538/.flair/datasets/common_characters_large'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        model = MNER(params, embeddings, pre_model)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        with torch.no_grad():
            acc, f1, prec, rec = evaluator.get_accuracy(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    main()
