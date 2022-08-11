import os

from dotenv import load_dotenv
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from mmnoise.datamodules import *
from mmnoise.models.retrieval import RetrievalModule
from mmnoise.utils import conf_resolvers  # registers custom resolver


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(link_to='model.opt_args')
        parser.add_lr_scheduler_args(link_to='model.lrsched_args')


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    load_dotenv('.env')

    cli = CLI(RetrievalModule, save_config_overwrite=True, parser_kwargs={'parser_mode': 'omegaconf'})