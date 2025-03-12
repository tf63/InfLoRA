import copy
import logging
import os
import os.path
import sys
import time

import torch
import wandb

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in [1993, 1994, 1995, 1996, 1997]:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    wandb.init(project='cil-inflora', name=args['config'], reinit=True, config=args, tags=[])

    if args['model_name'] in [
        'InfLoRA',
        'InfLoRA_domain',
        'InfLoRAb5_domain',
        'InfLoRAb5',
        'InfLoRA_CA',
        'InfLoRA_CA1',
    ]:
        logdir = 'logs/{}/{}_{}_{}/{}/{}/{}/{}_{}-{}'.format(
            args['dataset'],
            args['init_cls'],
            args['increment'],
            args['net_type'],
            args['model_name'],
            args['optim'],
            args['rank'],
            args['lamb'],
            args['lame'],
            args['lrate'],
        )
    else:
        logdir = 'logs/{}/{}_{}_{}/{}/{}'.format(
            args['dataset'], args['init_cls'], args['increment'], args['net_type'], args['model_name'], args['optim']
        )

    logfilename = os.path.join(logdir, '{}'.format(args['seed']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[logging.FileHandler(filename=logfilename + '.log'), logging.StreamHandler(sys.stdout)],
    )
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args['dataset'],
        args['shuffle'],
        args['seed'],
        args['init_cls'],
        args['increment'],
        data_dir='/dataset',
        args=args,
    )
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)

    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))

        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()

        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()

        # task_total = model.task_total
        inference_time = time_end - time_start
        # inference_time_per_data = inference_time / task_total
        # inference_time_total += inference_time

        # raise Exception
        model.after_task()

        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
        cnn_curve_task['top1'].append(cnn_accy_task)
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
        logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))

        data = {
            'class': (task + 1) * args['increment'],
            'cnn_average_acc': sum(cnn_curve['top1']) / len(cnn_curve['top1']),
            'cnn_top1': cnn_accy['top1'],
            'inference_time': inference_time,
            # 'inference_time_per_data': inference_time_per_data,
            # 'inference_time_total': inference_time_total,
        }
        wandb.log(data)


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


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
