import os

import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from src.hydra_runner import hydra_runner
from src.embed import generate_passage_embeddings
from src.index import build_index, add_to_index
from src.search import search_topk
from src.exact_rerank import exact_rerank_topk


@hydra_runner(config_path="conf", config_name="default")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.tasks.datastore.get('embedding', False):
        logging.info("\n\n************** Building Embedding ***********")
        generate_passage_embeddings(cfg)
    
    if cfg.tasks.datastore.get('index', False):
        logging.info("\n\n************** Indexing ***********")
        build_index(cfg)

    if cfg.tasks.datastore.get('add_to_index', False):
        logging.info("\n\n************** Adding Indexing ***********")
        add_to_index(cfg)
    
    if cfg.tasks.eval.get('search', False):
        logging.info("\n\n************** Running Search ***********")
        search_topk(cfg)

    if cfg.tasks.eval.get('exact_rerank', False):
        logging.info("\n\n************** Exact Search ***********")
        exact_rerank_topk(cfg)
    

if __name__ == '__main__':
    main()
