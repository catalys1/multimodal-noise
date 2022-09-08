from collections import Counter
from pathlib import Path
import re


__all__ = [
    'extract_wandb_metrics',
    'get_wandb_run_from_id',
]


def extract_wandb_metrics(logfile):
    '''Returns a list of dictionaries containing the logged metrics stored in the wandb binary
    log file. Each dict contains the metric values logged at a given trainer/glboal_step.
    '''
    data = Path(logfile).open('rb').read().decode('utf8', errors='ignore')
    # one byte indicating the length of the key, followed by the key, followed by the byte \x01 and then another
    # byte indicating the length of the value, followed by the numeric value (which could be an integer, a float
    # in normal notation, or a float in exponential notation
    regex = r'^[\x01-\xff]?([\d\w/@_-]+)\x01[\x01-\xff]([-+]?\d+\.?\d*(?:e-\d+)?).*?$'
    metrics = re.findall(regex, data, re.MULTILINE)
    # find global_steps
    step_idx = [i for i in range(len(metrics)) if metrics[i][0] == 'trainer/global_step'] + [None]
    key_occur = Counter(x[0] for x in metrics[step_idx[0]:])
    keys = []
    for k, n in key_occur.items():
        if n > 1: keys.append(k)
    keys = sorted(keys)
    # keys = sorted(set(x[0] for x in metrics[step_idx[0]:]))
    rows = []
    last_d = {}
    last_step = 0
    for i in range(len(step_idx) - 1):
        rng = slice(step_idx[i], step_idx[i + 1])
        step = int(metrics[step_idx[i]][1])
        if step == last_step:
            d = last_d
        else:
            d = {k: float('nan') for k in keys}
            rows.append(d)
        for k, v in metrics[rng]:
            if k in d: d[k] = float(v) if ('.' in v or 'e' in v) else int(v)
        last_d = d
        last_step = step
    return rows


def get_wandb_run_from_id(wandb_runs, wandb_id):
    for run in wandb_runs:
        if wandb_id in str(run):
            return run
    raise RuntimeError(f'No run with id "{wandb_id}" found in the provided list of runs')
