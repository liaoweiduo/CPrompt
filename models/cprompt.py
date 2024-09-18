
import logging
import copy
import numpy as np
import random
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from network.cprompt_net import CPrompt_Net
from utils.toolkit import target2onehot, tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.toolkit import count_parameters
from .base_learner import BaseLearner
import os
from scipy import stats

dataset_classes = {
    "cifar100_vit": 100,
    "cgqa": 100,
    "cobj": 30,
    "domainnet": 200,
    "imagenetr": 200,
    "stanfordcars":196
}

class CPrompt(BaseLearner):
    def __init__(self, args):
        self.args=args
        self.topk=args["topk"]
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._device = args['device'][0]
        self.dataset_name=args["dataset"]
        self.args["num_classes"] = dataset_classes.get(self.dataset_name, 0) 
        self._network=CPrompt_Net(self.args)
        self.acc=[]
        self.faa_accuracy_table=[]
        
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        model_save_dir = (self.args["root"] + '/' + self.args['log_name'] +
                          '/models/seed-' + str(self.args['seed']) + '/task-' + str(self._cur_task) + '/')
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        cur_task_nbclasses=data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_task_nbclasses
        self._network.update_fc(self._total_classes,cur_task_nbclasses)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=None)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=8)
        self._train(self.train_loader,self.test_loader)

        # save model
        self._save_model(model_save_dir)

        self._network.fix_branch_layer()
        
    def _train(self,train_loader,test_loader):
        self._network.to(self._device)
        
        enabled = set()
        enabled_params = []
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                if self.args['only_learn_slot'] and 'slot_attn' not in name:
                    # only train slot_attn
                    continue

                enabled.add(name)
                enabled_params.append(param)
        print(f"Parameters to be updated: {enabled}")

        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,lr=self.args["lr"],weight_decay=self.args["weight_decay"])
        optimizer = optim.SGD(enabled_params, momentum=0.9, lr=self.args["lr"], weight_decay=self.args["weight_decay"])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["epochs"])
        if self.args['only_learn_slot']:
            self._learn_slot(train_loader,test_loader,optimizer,scheduler)
        else:
            self._classifier_train(train_loader,test_loader,optimizer,scheduler)

    def _learn_slot(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            # correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                slots, attn, recon_loss = self._network.slot_forward(inputs)
                loss = recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # _, preds = torch.max(logits, dim=1)
                # correct += preds.eq(new_targets.expand_as(preds)).cpu().sum()
                # total += len(targets)

            scheduler.step()
            # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                self._cur_task, epoch + 1, self.args["epochs"], losses / len(train_loader))

            prog_bar.set_description(info)
        logging.info(info)

    def _classifier_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                new_targets=targets-self._known_classes
                logits,features = self._network.aux_forward(inputs)
                # logging.info(f"logits {logits.shape}: {logits[0].detach().cpu().numpy()}")
                # logging.info(f"targets {new_targets.shape}: {new_targets}")
                loss_aux=F.cross_entropy(logits,new_targets)
                loss=loss_aux
                
                if self._cur_task>0:
                    for k in range(self._cur_task):
                        old_logit=self._network.clas_w[k](features)['logits']
                        c1_logits=self._network.clas_w[self._cur_task](features)['logits']
                        bool_=torch.max(c1_logits,dim=1)[0]>torch.max(old_logit,dim=1)[0]+self.args["margin"]
                        t=torch.ones((bool_.shape)).to(self._device)
                        t[bool_==False]=self.args["tau"]
                        t=t.unsqueeze(1).repeat(1,self._total_classes - self._known_classes)
                        # t=t.unsqueeze(1).repeat(1,self.args["increment"])
                        ground=F.softmax(old_logit/t,dim=1).detach().clone()
                        loss_ccl = -torch.sum(ground * torch.log(F.softmax(old_logit,dim=1)), dim=1).mean()
                        loss+=self.args["alpha"]*loss_ccl/self._cur_task
                        
                gen_p=[]
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
                K=self._network.keys

                s, f = self._known_classes, self._total_classes
                # s=self._cur_task*self.args["increment"]
                # f=(self._cur_task+1)*self.args["increment"]
                if self._cur_task==0:
                    K = K[s:f]
                else:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1)
                mk = torch.einsum('bd,kd->bk', q, n_K)      # 只有10维

                # logging.info(f"mk {mk.shape}: {mk[0].detach().cpu().numpy()}")
                # logging.info(f"targets {targets.shape}: {targets}")

                loss_mk=F.cross_entropy(mk,targets)
                loss+=loss_mk
                
                m=torch.randint(0,self._cur_task+1,(len(mk),1))     # random select prompt for each sample

                if 'slot' in self.args['model_name'].lower():
                    slots, attn, _ = self._network.slot_forward(inputs)     # slots [bs, k, h]
                    # each image with its slot forward on a single slot2prompt model based on m
                    prompts_1 = torch.cat([self._network.ts_prompts_1[j](slots[i].unsqueeze(0)) for i, j in enumerate(m)], dim=0)  # [bs, l, d]
                    prompts_2 = torch.cat([self._network.ts_prompts_2[j](slots[i].unsqueeze(0)) for i, j in enumerate(m)], dim=0)  # [bs, l, d]
                    gen_p.append(prompts_1)
                    gen_p.append(prompts_2)
                else:
                    ts_prompts_1=self._network.ts_prompts_1
                    P1=torch.cat([ts_prompts_1[j].weight.unsqueeze(0) for j in m],dim=0)
                    gen_p.append(P1)
                    ts_prompts_2=self._network.ts_prompts_2
                    P2=torch.cat([ts_prompts_2[j].weight.unsqueeze(0) for j in m],dim=0)
                    gen_p.append(P2)
                out_gen=self._network(inputs,gen_p,train=True)
                loss_ce=F.cross_entropy(out_gen,new_targets)
                loss+=loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(new_targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), train_acc)
            
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader): 
        faa_y_true=[]
        total = 0

        cor=0
        faa_pred=[]

        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            gen_p=[]
            with torch.no_grad():
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
            
            K=self._network.keys
            
            f=self._total_classes
            # f=(self._cur_task+1)*self.args["increment"]
            K = K[:f]
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)

            if self._cur_task == 0:
                m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["init_cls"]
            else:
                m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["increment"]

            if self.args['debug']:
                logging.info(f'DEBUG: mk {mk.shape}: {mk[0]}')
                logging.info(f'DEBUG: m {m.shape}: {m[0]}')

            if 'slot' in self.args['model_name'].lower():
                with torch.no_grad():
                    slots, attn, _ = self._network.slot_forward(inputs)     # slots [bs, k, h]
                    # each image with its slot forward on a single slot2prompt model based on m
                    prompts_1 = torch.cat([self._network.ts_prompts_1[j](slots[i].unsqueeze(0)) for i, j in enumerate(m)], dim=0)  # [bs, l, d]
                    prompts_2 = torch.cat([self._network.ts_prompts_2[j](slots[i].unsqueeze(0)) for i, j in enumerate(m)], dim=0)  # [bs, l, d]
                gen_p.append(prompts_1)
                gen_p.append(prompts_2)
            else:
                ts_prompts_1=self._network.ts_prompts_1
                P1=torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
                gen_p.append(P1)
                ts_prompts_2=self._network.ts_prompts_2
                P2=torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
                gen_p.append(P2)
            
            with torch.no_grad():
                out_logits=self._network(inputs,gen_p,train=False)
            
            preds=torch.max(out_logits, dim=1)[1]
            
            logits_preds=torch.max(out_logits, dim=1)[1]
            cor+=preds.eq(targets.expand_as(preds)).cpu().sum().numpy()
            predicts = torch.topk(out_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            faa_pred.append(preds.cpu().numpy())
            faa_y_true.append(targets.cpu().numpy())
            
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            total+=len(targets)
        faa_pred=np.concatenate(faa_pred)
        faa_y_true=np.concatenate(faa_y_true)
        faa_tempacc=[]
        for class_id in range(0, np.max(faa_y_true), self.args["increment"]):
            idxes = np.where(np.logical_and(faa_y_true >= class_id, faa_y_true < class_id + self.args["increment"]))[0]
            faa_tempacc.append(np.around((faa_pred[idxes] == faa_y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        
        self.faa_accuracy_table.append(faa_tempacc)
        
        acctable = np.zeros([self._cur_task + 1, self._cur_task + 1])

        for idxx, line in enumerate(self.faa_accuracy_table):
            idxy = len(line)
            acctable[idxx, :idxy] = np.array(line)
        
        acctable = acctable.T
        
        forgetting = np.mean((np.max(acctable, axis=1) - acctable[:, self._cur_task])[:self._cur_task])
        
        self.acc.append(np.around(cor*100 / total, decimals=2))
        print("######################################")
        print("Last-acc:{}".format(self.acc[-1]))
        print("Avg-acc:{:.3f}".format(np.mean(self.acc)))
        print("FF: {}".format(np.around(forgetting, decimals=2)))
        print("test acc:{}".format(self.acc))
        print("FF table Last:{}".format(acctable[:,-1]))
        print("FF table:")
        print(acctable)
        print("################## next run ####################")
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def normal_eval_cnn(self,loader):
        self._network.eval()
        
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)
                
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def case_study(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        model_save_dir = (self.args["root"] + '/' + self.args['log_name'] +
                          '/models/seed-' + str(self.args['seed']) + '/task-' + str(self._cur_task) + '/')
        # if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        cur_task_nbclasses = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_task_nbclasses
        self._network.update_fc(self._total_classes, cur_task_nbclasses)
        logging.info('Case study on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))

        # train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
        #                                          mode='train', appendent=None)
        # self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=8,
        #                                persistent_workers=True, pin_memory=True)

        # shuffle=True, to see different targets in 1 batch and the same samples after different tasks.

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # test_datasets = {}
        self.test_loaders = []
        class_from = 0
        class_to = 0
        for task_id in range(self._cur_task + 1):
            task_nbclasses = data_manager.get_task_size(task_id)
            class_to = class_to + task_nbclasses
            test_dataset = data_manager.get_dataset(np.arange(class_from, class_to), source='test', mode='test')
            class_from = class_to
            test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=8)
            self.test_loaders.append(test_loader)

        # self._train(self.train_loader, self.test_loader)
        #
        # # save model
        # self._save_model(model_save_dir)

        self._load_model(model_save_dir)

        self._network.to(self._device)

        self._network.fix_branch_layer()

        ## start case study
        logging.info("######################################")
        self._network.eval()

        for task_id in range(self._cur_task + 1):
            logging.info(f"Eval on task: {task_id} ##########")

            loader = self.test_loaders[task_id]

            iterator = iter(loader)
            sample = next(iterator)

            _, inputs, targets = sample

            inputs, targets = inputs.to(self._device), targets.to(self._device)

            # logging.info(f'inputs {inputs.shape}: {inputs[:5,0,0]}')
            logging.info(f'targets {targets.shape}: {targets[:5]}')

            gen_p = []
            with torch.no_grad():
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:, 0, :]

            K = self._network.keys

            f = self._total_classes
            # f=(self._cur_task+1)*self.args["increment"]
            K = K[:f]
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)

            if self._cur_task == 0:
                m = torch.max(mk, dim=1, keepdim=True)[1] // self.args["init_cls"]
            else:
                m = torch.max(mk, dim=1, keepdim=True)[1] // self.args["increment"]

            logging.info(f'mk {mk.shape}: {mk[:5]}')
            logging.info(f'm {m.shape}: {m[:5, 0]}')       # [bs, 1]

            ts_prompts_1 = self._network.ts_prompts_1
            P1 = torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m], dim=0)
            gen_p.append(P1)
            ts_prompts_2 = self._network.ts_prompts_2
            P2 = torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m], dim=0)
            gen_p.append(P2)

            with torch.no_grad():
                out_logits = self._network(inputs, gen_p, train=False)

            preds = torch.max(out_logits, dim=1)[1]

            logging.info(f'out_logits {out_logits.shape}: {out_logits[:5]}')
            logging.info(f'preds {preds.shape}: {preds[:5]}')

        logging.info("################## next run ####################")
