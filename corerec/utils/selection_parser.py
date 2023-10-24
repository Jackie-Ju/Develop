import pandas as pd
import argparse
import os.path
from corerec.config.config_utils import load_config_data
from corerec.utils.utils import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--coreset_size', type=float, default=0.3, help="coreset size")
    parser.add_argument('--method', type=str, default="gsp", help="the method used to select coreset")
    parser.add_argument('--model', type=str, default="specwalk", help="the proxy model")
    parser.add_argument('--dataset', type=str, default="ml100k", help="the dataset to be selected")
    parser.add_argument('--setting', type=str, default="_debug",
                        help="post-fix of the method according to specific settings")
    parser.add_argument('--batch_size', type=int, default=512, help="the batch size")
    parser.add_argument('--device', type=str, default="cpu", help="the device to use")
    parser.add_argument('--epochs', type=int, default=5, help="number of training epochs")
    parser.add_argument('--T', type=int, default=200, help="scheduler")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--selection_type', type=str, default="user", help="only user selection is supported")
    parser.add_argument('--repeat', type=int, default=1, help="number of experiments")
    parser.add_argument('--seed', type=int, default=2020, help="the random seed")
    parser.add_argument('--eval_every', type=int, default=1,
                        help="evaluating and logging the data every n epochs")
    return parser.parse_args()


args = parse_args()
global_path = 'corerec/config/selection_global_config.py'
model_config_file = f'corerec/config/models/{args.model.strip()}_config.py'
strategy_config_file = f'corerec/config/strategies/{args.method.strip()}_config.py'
cfg = load_config_data(global_path, model_config_file, strategy_config_file)
cfg.train_args.device = args.device.strip()
cfg.dss_args.coreset_size = args.coreset_size
cfg.dataset.name = args.dataset.strip()
cfg.dataloader.batch_size = args.batch_size
cfg.train_args.num_epochs = args.epochs
cfg.scheduler.T_max = args.T
cfg.optimizer.lr = args.lr
cfg.dss_args.selection_type = args.selection_type.strip()
cfg.dss_args.repeat = args.repeat
cfg.dss_args.seed = args.seed
cfg.eval_args.eval_every = args.eval_every

if args.setting is not None:
    cfg.dss_args.setting = args.setting.strip()
    method_type = f"{cfg.dss_args.method}{cfg.dss_args.setting}"
else:
    method_type = f"{cfg.dss_args.method}"
cfg.method_type = method_type
results_dir = os.path.abspath(os.path.expanduser(cfg.train_args.results_dir))

result_dir = os.path.join(results_dir, method_type, cfg.dataset.name)
ensure_dir(result_dir)

cfg.eval_args.model_dict_path = os.path.join(cfg.early_stopping.root,
             f"{cfg.method_type}/{cfg.method_type}-{cfg.model.architecture}-"
             f"{cfg.dataset.name}-{cfg.dss_args.coreset_size}.pth")

logo = r'''
   ___     ___     ___     ___     ___     ___     ___   
  / __|   / _ \   | _ \   | __|   | _ \   | __|   / __|  
 | (__   | (_) |  |   /   | _|    |   /   | _|   | (__   
  \___|   \___/   |_|_\   |___|   |_|_\   |___|   \___|  
_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""| 
"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-' 
'''
print(logo)