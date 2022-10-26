import os
import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import loader
from model.model import MyModel
from utils import seed_worker, seed_everything, train, evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--dataset', type=str, default='twitter2017', choices=['twitter2015', 'twitter2017'])
parser.add_argument('--encoder_t', type=str, default='bert-base-uncased',
                    choices=['bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--encoder_v', type=str, default='', choices=['', 'resnet101', 'resnet152'])
parser.add_argument('--stacked', action='store_true', default=False)
parser.add_argument('--rnn',   action='store_true',  default=False)
parser.add_argument('--crf',   action='store_true',  default=False)
parser.add_argument('--aux',   action='store_true',  default=False)
parser.add_argument('--gate',   action='store_true',  default=False)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'AdamW'])
args = parser.parse_args()


if (args.aux or args.gate) and args.encoder_v == '':
    raise ValueError('Invalid setting: auxiliary task or gate module must be used with visual encoder (i.e. ResNet)')

seed_everything(args.seed)
generator = torch.Generator()
generator.manual_seed(args.seed)

if args.num_workers > 0:
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
ner_corpus = loader.load_ner_corpus(f'resources/datasets/{args.dataset}', load_image=(args.encoder_v != ''))
ner_train_loader = DataLoader(ner_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                              shuffle=True, worker_init_fn=seed_worker, generator=generator)
ner_dev_loader = DataLoader(ner_corpus.dev, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
ner_test_loader = DataLoader(ner_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)
if args.aux:
    itr_corpus = loader.load_itr_corpus('resources/datasets/relationship')
    itr_train_loader = DataLoader(itr_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                                  shuffle=True, worker_init_fn=seed_worker, generator=generator)
    itr_test_loader = DataLoader(itr_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)

model = MyModel.from_pretrained(args)

params = [
    {'params': model.encoder_t.parameters(), 'lr': args.lr},
    {'params': model.head.parameters(), 'lr': args.lr * 100},
]
if args.encoder_v:
    params.append({'params': model.encoder_v.parameters(), 'lr': args.lr})
    params.append({'params': model.proj.parameters(), 'lr': args.lr * 100})
if args.rnn:
    params.append({'params': model.rnn.parameters(), 'lr': args.lr * 100})
if args.crf:
    params.append({'params': model.crf.parameters(), 'lr': args.lr * 100})
if args.gate:
    params.append({'params': model.aux_head.parameters(), 'lr': args.lr * 100})
optimizer = getattr(torch.optim, args.optim)(params)

print(args)
dev_f1s, test_f1s = [], []
ner_losses, itr_losses = [], []
best_dev_f1, best_test_report = 0, None
for epoch in range(1, args.num_epochs + 1):
    if args.aux:
        itr_loss = train(itr_train_loader, model, optimizer, task='itr', weight=0.05)
        itr_losses.append(itr_loss)
        print(f'loss of image-text relation classification at epoch#{epoch}: {itr_loss:.2f}')

    ner_loss = train(ner_train_loader, model, optimizer, task='ner')
    ner_losses.append(ner_loss)
    print(f'loss of multimodal named entity recognition at epoch#{epoch}: {ner_loss:.2f}')

    dev_f1, dev_report = evaluate(model, ner_dev_loader)
    dev_f1s.append(dev_f1)
    test_f1, test_report = evaluate(model, ner_test_loader)
    test_f1s.append(test_f1)
    print(f'f1 score on dev set: {dev_f1:.4f}, f1 score on test set: {test_f1:.4f}')
    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_test_report = test_report

print()
print(best_test_report)

results = {
    'config': vars(args),
    'dev_f1s': dev_f1s,
    'test_f1s': test_f1s,
    'ner_losses': ner_losses,
    'itr_losses': itr_losses,
}
file_name = f'log/{args.dataset}/bs{args.bs}_lr{args.lr}_seed{args.seed}.json'
with open(file_name, 'w') as f:
    json.dump(results, f, indent=4)
