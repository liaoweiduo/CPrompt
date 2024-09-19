import sys
import os
import random
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

class Logger(object):
    def __init__(self, name, mode='a'):
        self.terminal = sys.stdout
        self.log = open(name, mode)

    def write(self, message):
        self.terminal.write(f"{message}")
        self.log.write(f"{message}")
        # self.log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {message}")

    def flush(self):
        self.log.flush()

def train(args):
    seed_list = copy.deepcopy(args['seed']) 
    device = copy.deepcopy(args['device'])

    if args['only_learn_slot']:
        args['log_name'] = args['slot_log_name']
        args['epochs'] = args['slot_epochs']

    # duplicate output stream to output file
    if not os.path.exists(args["root"] + '/' + args['log_name']): os.makedirs(args["root"] + '/' + args['log_name'])
    log_out = args["root"] + '/' + args['log_name'] + '/output.log'
    sys.stdout = Logger(log_out)
    log_err = args["root"] + '/' + args['log_name'] + '/err.log'
    sys.stderr = Logger(log_err, 'w')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):
    if not os.path.exists('./logs'): os.makedirs('./logs')
    logfilename = './logs/{}_{}_{}_{}_{}_{}'.format( args['log_name'], args['seed'], args['model_name'],
                                                     args['dataset'], args['init_cls'], args['increment'])
    file_handler = logging.FileHandler(logfilename)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s [%(filename)s] => %(message)s',
        handlers=[file_handler, console_handler],
        force=True,
    )

    _set_random()
    _set_device(args)
    print_args(args)
    
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)
    cnn_curve, nme_curve, nme_curve_ori = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))

        if not args['debug']:
            model.incremental_train(data_manager)
        
            cnn_accy,nme_accy = model.eval_task()
            model.after_task()
            if nme_accy is not None:
                logging.info('NME: {}'.format(nme_accy['grouped']))

                cnn_curve['top1'].append(cnn_accy['top1'])
                cnn_curve['top5'].append(cnn_accy['top5'])

                nme_curve['top1'].append(nme_accy['top1'])
                nme_curve['top5'].append(nme_accy['top5'])

                print('{}'.format(cnn_curve['top1']))
                print('{}'.format(cnn_curve['top5']))
                print('{}'.format(nme_curve['top1']))
                print('{}'.format(nme_curve['top5']))
                print('old:{}, new:{}'.format(nme_accy['grouped']['old'],nme_accy['grouped']['new']))
                x=np.array(cnn_curve['top1'])
                if len(x)>=1:
                    print("TLO:{}".format(x[-1]))
                    print("MEAN:{}".format(round(np.mean(x),2)))
            else:
                logging.info('No NME accuracy.')
                logging.info('CNN: {}'.format(cnn_accy['grouped']))

                cnn_curve['top1'].append(cnn_accy['top1'])

                logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
                print('old:{}, new:{}'.format(cnn_accy['grouped']['old'],cnn_accy['grouped']['new']))
                x=np.array(cnn_curve['top1'])
                if len(x)>=1:
                    print("TLO:{}".format(x[-1]))
                    print("MEAN:{}".format(round(np.mean(x),2)))
        else:
            model.case_study(data_manager)
            model.after_task()

    print("###################### next setting ######################")

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)
    args['device'] = gpus

def _set_random():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
