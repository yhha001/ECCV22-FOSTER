import sys
import os
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
 
    logfilename = 'logs/{}/{}/{}/{}/{}_{}_{}_{}_{}'.format(args["model_name"],args["dataset"], init_cls, args['increment'], args['prefix'], args['seed'], args['convnet_type'],
                                                 args['beta1'],args["beta2"])
    replay_dir = os.path.join(logs_name, 'replay_buffers')
        

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args['dataset'],
        args['shuffle'],
        args['seed'],
        args['init_cls'],
        args['increment'],
        args.get('data_path'),
        args.get('forget_classes_path'),
        args.get('move_forget_classes_to_end', False),
        args.get('exclude_forget_classes_from_schedule', False),
    )

    model = factory.get_model(args['model_name'], args)
    start_task = _restore_from_checkpoint_if_requested(model, data_manager, args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(start_task, data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        if args.get('export_replay_buffer', False):
            replay_path = os.path.join(
                replay_dir,
                '{}_task{:02d}.pt'.format(os.path.basename(logfilename), task),
            )
            model.export_replay_buffer(data_manager, replay_path)
        save_dir = os.path.join(
        "checkpoints",
        args["model_name"],
        args["dataset"],
        str(args.get("prefix", "default"))
        )
        os.makedirs(save_dir, exist_ok=True)

        net = model._network
        if hasattr(net, "module"):
            net = net.module

        snet = getattr(model, "_snet", None)
        if snet is not None and hasattr(snet, "module"):
            snet = snet.module

        ckpt = {
            "task": task,
            "cur_task": getattr(model, "_cur_task", task),
            "known_classes": model._known_classes,
            "total_classes": model._total_classes,
            "args": args,

            "network_state_dict": net.state_dict(),
            "snet_state_dict": snet.state_dict() if snet is not None else None,

            "data_memory": getattr(model, "_data_memory", None),
            "targets_memory": getattr(model, "_targets_memory", None),
            "class_means": getattr(model, "_class_means", None),
        }

        torch.save(ckpt, os.path.join(save_dir, f"task_{task}.pth"))

        if nme_accy is not None and cnn_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        elif nme_accy is None:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))
        else:
            logging.info('No CNN accuracy.')
            logging.info('NME: {}'.format(nme_accy['grouped']))

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))




def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _restore_foster_snet(model, checkpoint, args):
    snet_state = checkpoint.get('snet_state_dict')
    if snet_state is None:
        snet_state = checkpoint.get('network_state_dict')
    if snet_state is None:
        raise KeyError('Checkpoint does not contain snet_state_dict or network_state_dict.')

    total_classes = checkpoint['total_classes']
    snet = model._network.__class__(args['convnet_type'], False)
    snet.update_fc(total_classes)
    missing, unexpected = snet.load_state_dict(snet_state, strict=False)
    if missing or unexpected:
        logging.info(
            'Loaded resume network with missing keys: {} unexpected keys: {}'.format(
                missing, unexpected
            )
        )

    model._snet = snet
    model._network = snet
    model._network_module_ptr = snet


def _restore_from_checkpoint_if_requested(model, data_manager, args):
    checkpoint_path = args.get('resume_from_checkpoint') or args.get('resume_checkpoint')
    if not checkpoint_path:
        return 0

    if not os.path.isabs(checkpoint_path):
        cwd_checkpoint_path = os.path.abspath(checkpoint_path)
        repo_checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
        checkpoint_path = (
            cwd_checkpoint_path
            if os.path.isfile(cwd_checkpoint_path)
            else repo_checkpoint_path
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Resume checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if args.get('model_name') != 'foster':
        raise NotImplementedError('Checkpoint resume is currently implemented for FOSTER.')

    _restore_foster_snet(model, checkpoint, args)
    model._cur_task = checkpoint.get('cur_task', checkpoint['task'])
    model._known_classes = checkpoint['known_classes']
    model._total_classes = checkpoint['total_classes']
    model._data_memory = checkpoint.get('data_memory', model._data_memory)
    model._targets_memory = checkpoint.get('targets_memory', model._targets_memory)
    model._class_means = checkpoint.get('class_means', None)

    start_task = checkpoint['task'] + 1
    if start_task > data_manager.nb_tasks:
        raise ValueError(
            'Checkpoint task {} leaves no task to resume; dataset has {} tasks.'.format(
                checkpoint['task'], data_manager.nb_tasks
            )
        )
    logging.info(
        'Resumed FOSTER from {}; completed task {}, restarting at task {}.'.format(
            checkpoint_path, checkpoint['task'], start_task
        )
    )
    return start_task


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
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
