import pandas as pd
import argparse
import os.path
from corerec.config.config_utils import load_config_data
from corerec.utils.utils import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--coreset_size', type=float, default=0.3, help="coreset size")
    parser.add_argument('--method', type=str, default="Full", help="the method used to select coreset")
    parser.add_argument('--model', type=str, default="ease", help="the model to train")
    parser.add_argument('--dataset', type=str, default="ml100k", help="the original dataset")
    parser.add_argument('--setting', type=str, default="_debug",
                        help="post-fix of the method according to specific settings")
    parser.add_argument('--batch_size', type=int, default=512, help="the batch size")
    parser.add_argument('--device', type=str, default="cuda:1", help="the device to use")
    parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")
    parser.add_argument('--T', type=int, default=200, help="scheduler")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--repeat', type=int, default=1, help="number of experiments")
    parser.add_argument('--seed', type=int, default=2020, help="the random seed")
    parser.add_argument('--eval_every', type=int, default=1,
                        help="evaluating and logging the data every n epochs")
    parser.add_argument('--coreset_path', type=str, default='KCore_repeat1_0.5',
                        help="the path of file storing all the core users")
    return parser.parse_args()


args = parse_args()
global_path = 'corerec/config/recsys_global_config.py'
config_file = f'corerec/config/models/{args.model.strip()}_config.py'
cfg = load_config_data(global_path, config_file)
cfg.train_args.device = args.device.strip()
cfg.core_args.method = args.method.strip()
cfg.core_args.coreset = args.coreset_path
cfg.dataset.name = args.dataset.strip()
cfg.dataloader.batch_size = args.batch_size
cfg.train_args.num_epochs = args.epochs
cfg.scheduler.T_max = args.T
cfg.optimizer.lr = args.lr
cfg.dss_args.selection_type = args.selection_type.strip()
cfg.core_args.repeat = args.repeat
cfg.core_args.seed = args.seed
cfg.eval_args.eval_every = args.eval_every

# read in the coreset user ids and the random seed
if args.method.strip() != "Full" and args.coreset_path is not None:
    root_path = f"/mnt/recsys/zhengju/projects/CoreRec/short_paper/coresets/{cfg.dataset.name}/MF"
    file_path = os.path.join(root_path, f"{args.coreset_path}.csv")
    core_df = pd.read_csv(file_path)
    core_idx = list(core_df.userId.unique())
    cfg.dataset.core_user = core_idx
    cfg.core_args.seed = int(core_df.seed.values[0])
else:
    cfg.eval_args.coreset_affect=False
    
if args.setting is not None:
    cfg.core_args.setting = args.setting.strip()
    method_type = f"{cfg.core_args.type}{cfg.core_args.setting}"
else:
    method_type = f"{cfg.core_args.type}"
cfg.method_type = method_type
results_dir = os.path.abspath(os.path.expanduser(cfg.train_args.results_dir))

result_dir = os.path.join(results_dir, method_type, cfg.dataset.name)
ensure_dir(result_dir)

cfg.eval_args.model_dict_path = os.path.join(cfg.early_stopping.root,
             f"{cfg.method_type}/{cfg.method_type}-{cfg.model.architecture}-"
             f"{cfg.dataset.name}-{cfg.core_args.coreset_size}.pth")

logo = r'''
   ___     ___     ___     ___     ___     ___     ___   
  / __|   / _ \   | _ \   | __|   | _ \   | __|   / __|  
 | (__   | (_) |  |   /   | _|    |   /   | _|   | (__   
  \___|   \___/   |_|_\   |___|   |_|_\   |___|   \___|  
_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""|_|"""""| 
"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-' 
'''
print(logo)