import torch
import numpy as np
from argparse import ArgumentParser

from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP
from src.data.dataset import File_Dataset
from collections import defaultdict
import pandas as pd
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--distribution", type=str, help="distribution")
    parser.add_argument("--train_distribution", type=str, help="train distribution")
    parser.add_argument("--test_distribution", type=str, help="test distribution")
    # parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--checkpoint", type=str, default='best', help="Name of the checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    # parser.add_argument("--network_steps", type=int, default=100000, help="Number of network steps during evaluation")
    
    parser.add_argument("--network_steps", type=int, default=4000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=50, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print("CUDA Device {}: {}".format(i, device_name))

    if torch.cuda.is_available():
        if args.device is None:
            device = 'cuda:0' 
        else:
            device=f'cuda:{args.device}'

    else:
        device='cpu'
    

    name = 'model' if args.checkpoint is None else f'{args.checkpoint}'
    model_dir=f'pretrained agents/{args.train_distribution}'
    args.model_dir=model_dir
    model = ANYCSP.load_model(args.model_dir, name)
    model.eval()
    model.to(device)

    datapath=f'../data/testing/{args.test_distribution}'

    dataset = File_Dataset(datapath)

    num_solved = 0
    num_total = len(dataset)
    data_list = []
    df=defaultdict(list)
    for data in dataset:
        file = data.path
        max_val = data.constraints['ext'].cst_neg_mask.int().sum().cpu().numpy()
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')
        #with torch.cuda.amp.autocast():
        with torch.inference_mode():
            data = model(
                data,
                args.network_steps,
                return_all_assignments=False,
                return_log_probs=False,
                stop_early=True,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout,
            )

        best_per_run = data.best_num_unsat
        mean_best = best_per_run.mean()
        best = best_per_run.min().cpu().numpy()
        solved = best == 0
        num_solved += int(solved)
        best_cut_val = max_val - best

        print(
            f'{file}: {"Solved" if solved else "Unsolved"}, '
            f'Num Unsat: {int(best)}, '
            f'Cut Value: {best_cut_val}, '
            f'Steps: {data.num_steps}, '
            f'Opt Time: {data.opt_time:.2f}s, '
            f'Opt Step: {data.opt_step}'
        )

        df['cut'].append(best_cut_val)
        df['Opt Step'].append(data.opt_step)
        df['Opt Time'].append(data.opt_time)

    # print(f'Solved {100 * num_solved / num_total:.2f}%')
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(df)
    # df.sort_values(by=['File'])
    data_dir=f'generalization/{args.train_distribution}_ANYCSP'
    os.makedirs(data_dir,exist_ok=True)
    file_path=os.path.join(data_dir,f'results_{args.test_distribution}')
    df.to_pickle(file_path)
