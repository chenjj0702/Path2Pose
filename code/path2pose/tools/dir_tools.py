import os
import json
import csv
import datetime
import shutil
from pathlib import Path


class Log:
    def __init__(self, args, log_type, items):
        self.args = args
        self.type = log_type
        self.items = items

        t = datetime.datetime.now().strftime('%F_%H-%M-%S')
        self.file_name = os.path.join(self.args.save_dir, 'log_' + self.type + '_' + t + '.csv')
        with open(self.file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(items)

    def update(self, info):
        if len(info) > len(self.items):
            print('keys in info is more than that in log_items')
            print('keys in info ', info.keys())
            print('keys in log_items ', self.items)
            raise EOFError

        s = []
        for k in self.items:
            if k in info.keys():
                s.append(info[k])
            else:
                s.append(float('NaN'))

        with open(self.file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)


def save_args(args):
    """ 保存args参数为json """
    args_dict = vars(args)
    t = datetime.datetime.now().strftime('%F_%H-%M-%S')
    json_name = os.path.join(args.save_dir, 'params_' + t + '.json')
    with open(json_name, 'w') as f:
        json.dump(args_dict, f, indent=4)


def build_dir(args):
    """ build directory according to mode """
    logger_train = []
    logger_val = []

    if args.reuse is not None:
        """ 继续训练 """
        assert args.load_model is None
        pass

    elif args.mode == 'train':
        assert args.load_model is None

        # build results dir
        cwd = Path.cwd()  # 程序主目录
        version = str(cwd.name)
        t = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        args.save_dir = Path(args.save_dir) / (version + '_' + args.describe + '_' + t)
        if not args.save_dir.exists():
            args.save_dir.mkdir()

        # save code file
        obj_dir = args.save_dir / cwd.name
        shutil.copytree(str(cwd), str(obj_dir))

        args.save_dir = str(args.save_dir)

        # save args
        save_args(args)
        print(args)

        # log
        logger_train = Log(args, log_type='train', items=args.log_train_items)
        logger_val = Log(args, log_type='val', items=args.log_val_items)

        # build dir
        args.checkpoints_dir = os.path.join(args.save_dir, 'model')
        if not os.path.exists(args.checkpoints_dir):
            os.mkdir(args.checkpoints_dir)

        args.val_dir = os.path.join(args.save_dir, 'val_in_train')
        if not os.path.exists(args.val_dir):
            os.mkdir(args.val_dir)

    elif args.mode == 'val':
        assert args.load_model is not None
        t = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        dir1 = args.load_model.split('/')[1]
        dir2 = args.load_model.split('/')[-1].split('.')[0]  # epoch
        args.val_dir = '../' + dir1 + '/val_post_' + t + '/' + dir2
        if not os.path.exists(args.val_dir):
            os.makedirs(args.val_dir)

    elif args.mode == 'test':
        assert args.load_model is not None
        t = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        dir1 = args.load_model.split('/')[1]
        dir2 = args.load_model.split('/')[-1].split('.')[0]
        args.test_dir = '../' + dir1 + '/test_' + t + '/' + dir2
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)

    else:
        raise EOFError

    return logger_train, logger_val
