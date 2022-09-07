import os
import subprocess

import dotenv
from omegaconf import OmegaConf
import shortuuid

from mmnoise.utils import next_run_path


__all__ = [
    'slurm_defaults',
    'generate_id',
    'create_slurm_batch_file',
    'combine_from_files',
    'sync_slurm_and_config',
    'setup_run_for_slurm',
    'setup_run_for_local',
    'launch_slurm_jobs',
]


# load custom environment variables defined in `.env` at the root of the project
dotenv.load_dotenv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/.env')

CONFIG_TEMPLATE = os.path.join(os.environ['WORKDIR'], 'mmnoise/configs/template.yaml')


def slurm_defaults():
    return dict(
        nodes = 1,
        gpus = 1,
        mem = 40,
        cpus_per_task = 6,
        time = 360,
    )


def generate_id(length: int = 8) -> str:
    # copied from wandb: https://github.com/wandb/wandb/blob/v0.13.0/wandb/util.py#L743
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def create_slurm_batch_file(
    command,
    nodes = 1,
    gpus = 1,
    mem = 40,
    cpus_per_task = 6,
    time = 360,
    root_dir = os.environ['WORKDIR'],
    output_dir='logs/slurm',
    name = None,
):
    '''Generate the contents of a bash file that can be used to submit a job to slurm through sbatch.

    Args:
        command: string containing the python command to be executed, e.g. "python train.py fit --config=path"
        nodes: number of nodes
        gpus: number of gpus per node, world size will be nodes * gpus
        mem: amount of memory to allocate in GB
        cpus_per_task: number of cpus to allocate per task (or per gpu)
        time: allocated maximum time for the job, can be an int representing minutes, or a time string
            like 06:00:00 (hours:minutes:seconds)
        root_dir: path to root directory where command should be executed from
        output_dir: path to directory where logs will be created
    
    Returns:
        str, the full content of the batch file
    '''
    out_file = os.path.join(output_dir, 'slurm', 'slurm-%j.out')
    err_file = os.path.join(output_dir, 'slurm', 'slurm-%j.err')

    sbatch_opts = [
        f'--nodes={nodes}',
        f'--gres=gpu:{gpus}',
        f'--ntasks-per-node={gpus}',
        f'--cpus-per-task={cpus_per_task}',
        f'--mem={mem}G',
        f'-t {time}',
        f'--chdir={root_dir}',
        f'--output={out_file}',
        f'--error={err_file}',
    ]
    if name:
        sbatch_opts.append(f'--job-name={name}')
    sbatch_opts = '\n'.join(f'#SBATCH {x}' for x in sbatch_opts)
    sbatch = (
        '#!/bin/bash\n'
        f'{sbatch_opts}\n\n'
        'source ~/.bashrc\n'
        'source .env\n'
        f'conda activate $CONDA_ENV_NAME\n\n'
        f'srun {command}\n'
    )
    return sbatch


def combine_from_files(*configs, **named_configs):
    '''Create a single config by loading and merging multiple config files. The named_configs
    get added as nodes using their name as the root key.
    '''
    configs = [OmegaConf.load(c) for c in configs]
    configs.extend([OmegaConf.create({k: OmegaConf.load(c)}) for k, c in named_configs.items()])
    config = OmegaConf.merge(*configs)
    return config


def sync_slurm_and_config(config, slurm_kw, conf_key, slurm_key):
    '''Ensures that the config and slurm options are in sync for a given key. Priority is given
    to config values: slurm values will be overwritten when a config value exists. Otherwise, the
    slurm value is injected into the config.
    
    conf_key should be given as dotlist path, like arg.segment.key.
    '''
    conf_val = OmegaConf.select(config, conf_key, default='__missing__')
    if conf_val != '__missing__':
        slurm_kw[slurm_key] = conf_val
    else:
        OmegaConf.update(config, conf_key, slurm_kw[slurm_key])
    

def setup_run_for_slurm(config, slurm_kw=None, log_dir=None):
    '''Sets everything up for launching a run. Syncs config and slurm parameters, creates the run directory
    and sets corresponding paths in the config, and saves the config and bash file for launching the job
    in the run directory.
    '''
    log_dir = log_dir or os.path.join(os.environ['WORKDIR'], 'logs')
    slurm_def = slurm_defaults()
    if slurm_kw:
        slurm_def.update(slurm_kw)
    slurm_kw = slurm_def

    # create the run log directory
    run_dir = next_run_path(log_dir)
    os.makedirs(run_dir, exist_ok=False)  # should be a new run dir
    os.makedirs(os.path.join(run_dir, 'slurm'))

    # update slurm and config with paths
    # log and checkpoint paths should be defined relative to trainer.default_root_dir using node interpolation
    slurm_kw['output_dir'] = run_dir
    slurm_kw['root_dir'] = os.environ['WORKDIR']
    slurm_kw['name'] = run_dir.rstrip('/').rsplit('/', 1)[-1]
    config.trainer.default_root_dir = run_dir

    # synchronize config and slurm args
    sync_slurm_and_config(config, slurm_kw, 'trainer.num_nodes', 'nodes')
    sync_slurm_and_config(config, slurm_kw, 'trainer.devices', 'gpus')
    sync_slurm_and_config(config, slurm_kw, 'data.init_args.num_workers', 'cpus_per_task')
    if slurm_kw['nodes'] > 1 or slurm_kw['gpus'] > 1:
        config.trainer.strategy = 'ddp'

    # inject a wandb run id
    wandb_id = generate_id()
    OmegaConf.update(config, 'trainer.logger.0.init_args.id', wandb_id)

    # save files to run dir
    config_path = os.path.join(run_dir, 'raw_run_config.yaml')
    OmegaConf.save(config, config_path)
    command = f'python train.py fit --config={config_path}'
    slurm_file_content = create_slurm_batch_file(command, **slurm_kw)
    job_file = os.path.join(run_dir, 'job_submit.sh')
    with open(job_file, 'w') as f:
        f.write(slurm_file_content)
    with open(os.path.join(run_dir, 'wandb_id'), 'w') as f:
        f.write(wandb_id)

    # return path to slurm job file
    return job_file


def setup_run_for_local(config):
    '''Sets up a run for local execution, without a job submission script.
    '''
    # create the run log directory
    run_dir = next_run_path(os.path.join(os.environ['WORKDIR'], 'logs'))
    os.makedirs(run_dir, exist_ok=False)  # should be a new run dir

    # update config with paths
    # log and checkpoint paths should be defined relative to trainer.default_root_dir using
    # OmegaConf node interpolation
    config.trainer.default_root_dir = run_dir

    if config.trainer.num_nodes > 1 or config.trainer.devices > 1:
        config.trainer.strategy = 'ddp'

    # inject a wandb run id
    wandb_id = generate_id()
    OmegaConf.update(config, 'trainer.logger.0.id', wandb_id)

    # save files to run dir
    config_path = os.path.join(run_dir, 'raw_run_config.yaml')
    OmegaConf.save(config, config_path)
    with open(os.path.join(run_dir, 'wandb_id'), 'w') as f:
        f.write(wandb_id)

    # return path to config for the run
    return config_path


def launch_slurm_jobs(job_file_list):
    '''Given a list of paths to slurm job submission bash files, execute each using sbatch
    to submit the jobs to the slurm scheduler.
    '''
    for job_file in job_file_list:
        args = ['sbatch', job_file]
        subprocess.run(args)
