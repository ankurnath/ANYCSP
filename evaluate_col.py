import torch
import numpy as np
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm
from src.data.dataset import File_Dataset
from collections import defaultdict

from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP
from src.data.dataset import nx_to_col,GraphDataset
from src.utils.data_utils import load_dimacs_graph
# from src.utils.dataset import 
import pandas as pd
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--distribution", type=str,help="Distribution of graphs")
    # parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--checkpoint_name", type=str, default='best', help="Name of the checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--network_steps", type=int, default=10000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=50, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    parser.add_argument("--num_colors", type=int,default=3, help="Number of colors")
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the number of CUDA devices
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

    # name = 'model' if args.checkpoint is None else f'{args.checkpoint}'
    model_dir=f'pretrained agents color/{args.distribution}'
    model = ANYCSP.load_model(model_dir, args.checkpoint_name)
    # model = ANYCSP.load_model(args.model_dir, name)
    model.eval()
    model.to(device)

    datapath=f'../data_color/testing/{args.distribution}'
    # dataset = File_Dataset(datapath)
    dataset=GraphDataset(datapath,ordered=True)

    # data_dict = {p: load_dimacs_graph(p) for p in tqdm(glob(args.data_path))}
    # data_dict = {p: nx_to_col(g, args.num_colors) for p, g in data_dict.items()}
    data_dict ={i: nx_to_col(dataset.get(), args.num_colors) for i in range(len(dataset))}

    num_solved = 0
    num_total = len(data_dict)
    df=defaultdict(list)
    for file, data in data_dict.items():
        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')
        # with torch.cuda.amp.autocast():
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

        print(
            f'{file}: {"Solved" if solved else "Unsolved"}, '
            f'Num Unsat: {int(best)}, '
            f'Steps: {data.num_steps}, '
            f'Opt Time: {data.opt_time:.2f}s, '
            f'Opt Step: {data.opt_step}'
        )

        df["Solved"].append(True if solved else False)
        df['Steps'].append(data.num_steps)
        df['Num Unsat'].append(int(best))
        df['Opt Step'].append(data.opt_step)
        df['Opt Time'].append(data.opt_time)

    print(f'Solved {100 * num_solved / num_total:.2f}%')

    df=pd.DataFrame(df)
    data_folder=os.path.join(model_dir,'data')
    os.makedirs(data_folder,exist_ok=True)
    df.to_pickle(os.path.join(data_folder,f'results_{args.network_steps}'))
    # print(f'Solved {100 * num_solved / num_total:.2f}%, Average Time: {total_time / num_total:.2f}s')
