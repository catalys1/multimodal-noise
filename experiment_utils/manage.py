"""Utilities for helping to manage experiment logs and files.
"""
from pathlib import Path
import subprocess

from .wandb_utils import get_wandb_run_from_id


__all__ = [
    'process_run_ids',
    'purge_runs',
    'sync_wandb_offline_runs',
]


def process_run_ids(run_ids):
    ids = []
    for rid in run_ids:
        if '-' in rid:
            s, e = rid.strip().split('-')
            ids.extend(list(range(int(s), int(e)+1)))
        else:
            ids.append(int(rid))
    ids = sorted(set(ids))
    return ids


def purge_runs(runs_path, run_ids, wandb_root='./wandb', dryrun=True):
    '''Delete logging information for a given set of runs without removing configs or job launch
    scripts.
    '''
    wandb_runs = list(Path(wandb_root).glob('offline-run-*'))
    if dryrun:
        print('Dryrun: the following files would be removed')
    run_ids = process_run_ids(run_ids)
    for n in run_ids:
        root = Path(f'{runs_path}/run-{n}')
        ckpt = root.joinpath('checkpoints')
        if ckpt.exists():
            for x in ckpt.iterdir():
                if dryrun:
                    print(str(x))
                else:
                    x.unlink()
            if not dryrun:
                ckpt.rmdir()
        slurm = root.joinpath('slurm')
        if slurm.exists():
            for x in slurm.iterdir():
                if dryrun:
                    print(str(x))
                else:
                    x.unlink()
        wid = root.joinpath('wandb_id').open().read().strip()
        wandb = get_wandb_run_from_id(wandb_runs, wid)
        if dryrun:
            print(str(wandb))
        else:
            subprocess.run(['rm', '-r', str(wandb)])


def sync_wandb_offline_runs(wandb_root='./wandb', runs_path=None, run_ids=None):
    '''Uses `wandb sync` to sync offline runs to the wandb server.
    
    Args:
        wandb_root: path to root wandb folder, where individual runs are stored.
        runs_path: (optional) path to root folder where run logs are stored, used in conjuction with `run_ids`
            to specify which runs to synchronize.
        run_ids: (optional) a list of integer run numbers to sync from. If runs_path and run_ids are specified,
            they will be used to decide which runs to sync. Otherwise, all unsynced runs under `wandb_root` will
            be synced.
    '''
    wandb_runs = list(Path(wandb_root).glob('offline-run-*'))
    to_sync = []
    if runs_path and run_ids:
        run_ids = process_run_ids(run_ids)
        for n in run_ids:
            run = Path(f'{runs_path}/run-{n}')
            wandb_id = run.joinpath('wandb_id').open().read().strip()
            wandb = get_wandb_run_from_id(wandb_runs, wandb_id)
            to_sync.append(str(wandb))
    else:
        for wandb in wandb_runs:
            wandb_id = wandb.name.rsplit('-', 1)[1]
            if wandb.joinpath(f'run-{wandb_id}.wandb.synced').exists():
                continue  # skip runs that have already been synced
            to_sync.append(str(wandb))
    print(f'Syncing {len(to_sync)} wandb runs')
    subprocess.run(['wandb', 'sync', '--no-include-synced', '--mark-synced'] + to_sync)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    p = subparsers.add_parser('purge')
    p.add_argument('--dryrun', type=t_or_f, default=True)
    p.add_argument('--runs_path', type=str, default='./rlogs')
    p.add_argument('--run_ids', type=str, nargs='+', required=True)
    p.add_argument('--wandb_root', type=str, default='./rlogs/wandb')
    p.set_defaults(func=purge_runs)

    p = subparsers.add_parser('sync')
    p.add_argument('--wandb_root', type=str, default='rlogs/wandb')
    p.add_argument('--runs_path', type=str, default=None)
    p.add_argument('--run_ids', type=int, nargs='*', default=None)
    p.set_defaults(func=sync_wandb_offline_runs)

    args = parser.parse_args()
    args = args.__dict__
    func = args.pop('func')
    func(**args)
