import torch.utils.data
from torchcrf import CRF
from model import MNER
from MM_pretrain.model import *
from timeit import default_timer as timer
from util import *
from tqdm import tqdm
import numpy as np
from flair.embeddings import *
from MM_pretrain.resnet_vlbert import ResNetVLBERT


def init_xavier(m):
    """
    Sets all the linear layer weights as per xavier initialization
    :param m:
    :return: Nothing
    """
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


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
                '/media/iot538/a73dbfc5-a8a0-4021-a841-3b7d7f3fd964/mnt/xj/wnut17_advanced/pretrain/en-fasttext-crawl-300d-1M'),
            CharacterEmbeddings('/home/iot538/.flair/datasets/common_characters_large'),
            # FlairEmbeddings('/home/iot538/.flair/embeddings/news-forward-0.4.1.pt'),
            # FlairEmbeddings('/home/iot538/.flair/embeddings/news-backward-0.4.1.pt'),
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
        # optimizer = torch.optim.SGD(ner_model.parameters(), lr=self.params.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
        # optimizer = torch.optim.Adam(ner_model.parameters(), lr=self.params.lr)

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
                    # relation_ = relation.detach()
                    # ner训练
                    emissions, attention_probs = ner_model(to_variable(x), x_flair, to_variable(x_obj),
                                          lens, to_variable(mask), relation.detach())  # seq_len * bs * labels
                    tags = to_variable(y).transpose(0, 1).contiguous()  # seq_len * bs
                    mask = to_variable(mask).byte().transpose(0, 1)  # seq_len * bs

                    # computing crf loss
                    loss = -loss_function(emissions, tags, mask=mask)
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())

                    # if self.params.clip_value > 0:
                    #     torch.nn.utils.clip_grad_norm(ner_model.parameters(), self.params.clip_value)
                    optimizer.step()

                scheduler.step()
                optim_state = optimizer.state_dict()
                # optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / (
                #             1 + self.params.gamma * (epoch + 1))
                # optimizer.load_state_dict(optim_state)
                torch.cuda.empty_cache()
                # Calculate accuracy and save best model
                if (epoch + 1) % self.params.validate_every == 0:
                    # acc_dev, f1_dev, p_dev, r_dev = self.evaluator.get_accuracy(model, 'val', loss_function)
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
                            print("model save in " + model_path)
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
                torch.cuda.empty_cache()
                if epoch + 1 == self.params.num_epochs:
                    best_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(best_epoch, prev_best)
                    print("{} epoch get the best f1 {:.5f}".format(best_epoch, prev_best))
                    print("the model is save in " + model_path)
        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')
