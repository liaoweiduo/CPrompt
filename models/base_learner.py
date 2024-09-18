import copy
import logging
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args=args

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
                                                   decimals=2)
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets, classnames) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)
                outputs=logits.softmax(dim=-1)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _save_model(self, filename):
        model_state = self._network.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()

        torch.save(model_state, filename + 'class.pth')
        logging.info('=> Saving class model to:' + filename)
        logging.info('=> Save Done')

    def _load_model(self, filename, drop_last=False):
        state_dict = torch.load(filename + 'class.pth')
        # complete with/without module.
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key[7:]] = state_dict[key]
            else:
                state_dict[f'module.{key}'] = state_dict[key]
        if drop_last:
            del state_dict['module.last.weight']; del state_dict['module.last.bias']
            del state_dict['last.weight']; del state_dict['last.bias']
            # if 'module.last.weight' in state_dict:
            #     del state_dict['module.last.weight']; del state_dict['module.last.bias']
            # else:
            #     del state_dict['last.weight']; del state_dict['last.bias']
            # self.model.load_state_dict(state_dict, strict=False)
        self._network.load_state_dict(state_dict, strict=False)
        logging.info('=> Load Done')

        self._network.eval()
