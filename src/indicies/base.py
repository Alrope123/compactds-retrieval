import os
import logging
import numpy as np
import torch

from src.indicies.flat import FlatIndexer
from src.indicies.ivf_flat import IVFFlatIndexer
from src.indicies.ivf_pq import IVFPQIndexer
from src.indicies.index_utils import get_index_dir_and_embedding_paths


class Indexer(object):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.datastore.index
        self.index_type = self.args.index_type
        
        passage_dir = self.cfg.datastore.embedding.passages_dir
        deprioritized_domains = self.args.get('deprioritized_domains', [])
        index_dir, embedding_paths = get_index_dir_and_embedding_paths(cfg, deprioritized_domains=deprioritized_domains)
        os.makedirs(index_dir, exist_ok=True)
        # logging.info(f"Indexing for passages: {embedding_paths}")
        if "IVF" in self.index_type:
            formatted_index_name = f"index_{self.index_type}.{self.args.sample_train_size}.{self.args.projection_size}.{self.args.ncentroids}.faiss"
            if "PQ" in self.index_type:
                formatted_index_name = formatted_index_name.replace(".faiss", f".{self.args.n_subquantizers}.faiss")
            trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
        else:
            formatted_index_name = f"index_{self.index_type}.faiss"
        index_path = os.path.join(index_dir, formatted_index_name)
        meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
        pos_array_save_path = os.path.join(index_dir, 'passage_pos_id_array.npy')
        passage_filenames_save_path = os.path.join(index_dir, 'passage_filenames.npy')

        sample_train_path = self.args.sample_train_path if "sample_train_path" in self.args else None
        save_intermediate_index = self.args.save_intermediate_index if "save_intermediate_index" in self.args else False
        
        if self.index_type == "Flat":
            self.datastore = FlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                passage_dir=passage_dir,
                pos_array_save_path=pos_array_save_path,
                passage_filenames_save_path=passage_filenames_save_path,
                dimension=self.args.projection_size,
            )
        elif self.index_type == "IVFFlat":
            self.datastore = IVFFlatIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                pos_array_save_path=pos_array_save_path,
                passage_filenames_save_path=passage_filenames_save_path,
                sample_train_size=self.args.sample_train_size,
                prev_index_path=None,
                dimension=self.args.projection_size,
                ncentroids=self.args.ncentroids,
                probe=self.args.probe,
            )
        elif self.index_type == "IVFPQ":
            self.datastore = IVFPQIndexer(
                embed_paths=embedding_paths,
                index_path=index_path,
                meta_file=meta_file,
                trained_index_path=trained_index_path,
                passage_dir=passage_dir,
                deprioritized_domains=deprioritized_domains,
                pos_array_save_path=pos_array_save_path,
                passage_filenames_save_path=passage_filenames_save_path,
                sample_train_size=self.args.sample_train_size,
                sample_train_path=sample_train_path,
                prev_index_path=None,
                save_intermediate_index=save_intermediate_index,
                dimension=self.args.projection_size,
                ncentroids=self.args.ncentroids,
                probe=self.args.probe,
                n_subquantizers=self.args.n_subquantizers,
                code_size=self.args.n_bits,
            )
        else:
            raise NotImplementedError
        
        
    def search(self, query_embs, k=5):
        all_scores, all_domains, all_passages, db_ids = self.datastore.search(query_embs, k)
        return all_scores, all_domains, all_passages, db_ids
    
    def add_to(self):
        import glob
        embedding_args = self.cfg.datastore.embedding
        index_args = self.cfg.datastore.index
        
        embedding_paths = glob.glob(index_args.new_passages_embeddings)
        print(f"new_passages_embeddings: {index_args.new_passages_embeddings}")
        def sort_func(x):
            domain = x.split("/")[-1].split(f'{embedding_args.prefix}')[0].split('--')[0] 
            rank, shard_idx = x.split("/")[-1].split(f'{embedding_args.prefix}')[-1].split(".pkl")[0].split("_")
            return domain, int(rank), int(shard_idx)
        embedding_paths = sorted(embedding_paths, key=sort_func)
        embedding_paths = embedding_paths if index_args.num_subsampled_embedding_files == -1 else embedding_paths[0:index_args.num_subsampled_embedding_files]

        self.datastore.add_new_embeddings(embedding_paths, embedding_args.new_passages_dir)