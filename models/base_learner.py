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
        if self.args['only_learn_slot']:
            losses, y_true = self._eval_cnn(self.test_loader)    # losses shape [bs, 1]
            cnn_losses = self._evaluate(losses, y_true, loss=True)
            return cnn_losses, None
        else:
            y_pred, y_true = self._eval_cnn(self.test_loader)
            cnn_accy = self._evaluate(y_pred, y_true)
            if hasattr(self, '_class_means'):
                y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                nme_accy = self._evaluate(y_pred, y_true)
            else:
                nme_accy = None

            return cnn_accy, nme_accy
    
    def _evaluate(self, y_pred, y_true, loss=False):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, loss=loss)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        if not loss:
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
        logging.info(f'=> Load from {filename}')
        state_dict = torch.load(filename + 'class.pth')
        # complete with/without module.
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key[7:]] = state_dict.pop(key)
            if drop_last and 'clas_w' in key:
                del state_dict[key]
        self._network.load_state_dict(state_dict, strict=True)
        logging.info(f'=> Load Done with params: {list(state_dict.keys())}')

        self._network.eval()
