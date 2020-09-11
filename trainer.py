import torch.utils.data
from torchcrf import CRF
from mner import MNER
from rpbert.bert_rel import *
from timeit import default_timer as timer
from util import *
from tqdm import tqdm
import numpy as np
from flair.embeddings import *



def burnin_schedule(i):
    if i < 10:
        factor = 1
    elif i < 20:
        factor = 0.1
    else:
        factor = 0.01
    return factor


class Trainer:
    def __init__(self, params, config, data_loader, dlbb, evaluator, pre_model=None):
        self.params = params
        self.config = config
        self.data_loader = data_loader
        self.dlbb = dlbb
        self.evaluator = evaluator
        self.pre_model = pre_model

    def train(self):
        num_of_tags = len(self.data_loader.labelVoc)
        embedding_types = [
            WordEmbeddings(
                'pretrained/embedding/en-fasttext-crawl-300d-1M'),
            CharacterEmbeddings('pretrained/embedding/datasets/common_characters_large'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        ner_model = MNER(self.params, embeddings, self.pre_model, num_of_tags=10)
        loss_function_relation = nn.CrossEntropyLoss()
        loss_function = CRF(num_of_tags)
        if torch.cuda.is_available():
            ner_model = ner_model.cuda()
            loss_function = loss_function.cuda()

        paras = dict(ner_model.named_parameters())
        paras_new = []
        for k, v in paras.items():
            if 'pre_resnet' in k or 'vlbert' in k:
                paras_new += [{'params': [v], 'lr': 1e-6}]
            else:
                paras_new += [{'params': [v], 'lr': 1e-4}]
        optimizer = torch.optim.Adam(paras_new, weight_decay=self.params.wdecay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

        try:
            prev_best = 0
            best_epoch = 0
            for epoch in range(self.params.num_epochs):
                losses = []
                start_time = timer()

                # relation任务
                for (x, x_obj, y, mask, lens, ifpairs) in tqdm(self.dlbb.train_phase1_loader):
                    ner_model.train()
                    optimizer.zero_grad()
                    pair_out = ner_model.pretrain_model(to_variable(x), to_variable(x_obj), lens,
                                     to_variable(mask))  # seq_len * bs * labels
                    ifpairs = to_variable(ifpairs).contiguous()
                    loss1 = loss_function_relation(pair_out.squeeze(1), ifpairs)
                    loss1.backward()
                    optimizer.step()

                torch.cuda.empty_cache()
                # ner任务
                for (x, x_flair, x_obj, y, mask, lens) in tqdm(
                        self.data_loader.train_data_loader):
                    ner_model.train()
                    pair_out = ner_model(to_variable(x), x_flair, to_variable(x_obj),
                                        lens, to_variable(mask))

                    relation = F.softmax(pair_out, dim=-1)

                    optimizer.zero_grad()

                    # ner训练
                    emissions, attention_probs = ner_model(to_variable(x), x_flair, to_variable(x_obj),
                                          lens, to_variable(mask), relation.detach())  # seq_len * bs * labels
                    tags = to_variable(y).transpose(0, 1).contiguous()  # seq_len * bs
                    mask = to_variable(mask).byte().transpose(0, 1)  # seq_len * bs

                    # computing crf loss
                    loss = -loss_function(emissions, tags, mask=mask)
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())


                    optimizer.step()

                scheduler.step()
                optim_state = optimizer.state_dict()

                torch.cuda.empty_cache()
                # Calculate accuracy and save best rpbert
                if (epoch + 1) % self.params.validate_every == 0:
                
                    with torch.no_grad():
                        acc_dev, f1_dev, p_dev, r_dev = self.evaluator.get_accuracy(ner_model, 'test', loss_function)

                        print(
                            "Epoch {} : Training Loss: {:.5f}, Acc: {:.5f}, F1: {:.5f}, Prec: {:.5f}, Rec: {:.5f}, LR: {:.5f}"
                            "Time elapsed {:.2f} mins"
                                .format(epoch + 1, np.asscalar(np.mean(losses)), acc_dev, f1_dev, p_dev, r_dev,
                                        optim_state['param_groups'][0]['lr'],
                                        (timer() - start_time) / 60))
                        if f1_dev > prev_best and f1_dev > 0.7:
                            print("f1-score increased....saving weights !!")
                            best_epoch = epoch + 1
                            prev_best = f1_dev
                            model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(epoch + 1, f1_dev)
                            torch.save(ner_model.state_dict(), model_path)
                            print("rpbert save in " + model_path)
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
                torch.cuda.empty_cache()
                if epoch + 1 == self.params.num_epochs:
                    best_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(best_epoch, prev_best)
                    print("{} epoch get the best f1 {:.5f}".format(best_epoch, prev_best))
                    print("the rpbert is save in " + model_path)
        except KeyboardInterrupt:
            print("Interrupted.. saving rpbert !!!")
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')
