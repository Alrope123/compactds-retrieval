import os
import json
import pickle
import time
from tqdm import tqdm
from typing import List

import faiss
import numpy as np
import torch
from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class IVFPQIndexer(object):

    def __init__(self, 
                embed_paths,
                index_path,
                meta_file,
                trained_index_path,
                deprioritized_domains=[],
                passage_dir=None,
                pos_array_save_path=None,
                passage_filenames_save_path=None,
                sample_train_size=1000000,
                sample_train_path=None,
                prev_index_path=None,
                save_intermediate_index=False,
                dimension=768,
                dtype=np.float16,
                ncentroids=4096,
                probe=2048,
                num_keys_to_add_at_a_time=1000000,
                DSTORE_SIZE_BATCH=51200000,
                n_subquantizers=16,
                code_size=8,
                ):
    
        self.embed_paths = embed_paths  # list of paths where saved the embedding of all shards
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.prev_index_path = prev_index_path  # add new data to it instead of training new clusters
        self.save_intermediate_index = save_intermediate_index
        self.trained_index_path = trained_index_path  # path to save the trained index
        self.passage_dir = passage_dir
        self.deprioritized_domains = deprioritized_domains
        self.pos_array_save_path = pos_array_save_path
        self.passage_filenames_save_path = passage_filenames_save_path

        self.sample_size = sample_train_size
        self.sample_path = sample_train_path
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        self.n_subquantizers = n_subquantizers
        self.code_size = code_size
        self.random_seed = 1234

        np.random.seed(self.random_seed)

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_file_id = self.load_index_id_to_file_id()
            self.index.nprobe = self.probe
        
        else:
            self.index_id_to_file_id = []
            if not os.path.exists(self.trained_index_path):
                print ("Training index...")
                self._sample_and_train_index()

            print ("Building index...")
            self.index = self._add_keys(self.index_path, self.prev_index_path if self.prev_index_path is not None else self.trained_index_path)
        
        if self.pos_array_save_path is not None:
            self.psg_pos_id_array, self.passage_filenames = self.load_psg_pos_id_array()

        print ("index:", self.index)
        print (self.index.ntotal)
        print(f"DEBUG: length of index_id_to_file_id: {len(self.index_id_to_file_id)}")
        print(f"DEBUG: length of psg_pos_id_array: {len(self.psg_pos_id_array) if hasattr(self, 'psg_pos_id_array') else 'N/A'}")
        print(f"DEBUG: length of passage_filenames: {len(self.passage_filenames) if hasattr(self, 'passage_filenames') else 'N/A'}")
    
    def load_index_id_to_file_id(self,):
        with open(self.meta_file, "rb") as reader:
            index_id_to_file_id = np.load(reader)
        return index_id_to_file_id

    '''
    def load_embeds(self, shard_id=None):
        all_ids, all_embeds = [], []
        offset = 0
        for embed_path in self.embed_paths:
            loaded_shard_id = int(re.search(r'_(\d+).pkl$', embed_path).group(1))
            if shard_id is not None and loaded_shard_id != shard_id:
                continue
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            all_ids.extend([i + offset for i in ids])
            all_embeds.extend(embeddings)
            offset += len(ids)
        all_embeds = np.stack(all_embeds).astype(np.float32)
        datastore_size = len(all_ids)
        return all_embeds
        
    def get_embs(self, indices=None, shard_id=None):
        if indices is not None:
            embs = self.embs[indices]
        elif shard_id is not None:
            embs = self.load_embeds(shard_id)
        return embs

    def get_knn_scores(self, query_emb, indices):
        assert self.embeds is not None
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores
    '''

    def _sample_and_train_index(self,):
        if self.sample_path is None or not os.path.exists(self.sample_path):
            print(f"Sampling {self.sample_size} examples from {len(self.embed_paths)} files...")
            per_shard_sample_size = self.sample_size // len(self.embed_paths)
            all_sampled_embs = []
            for embed_path in self.embed_paths:
                print(f"Loading pickle embedding from {embed_path}...")
                with open(embed_path, "rb") as fin:
                    _, embeddings = pickle.load(fin)
                shard_size = len(embeddings)
                print(f"Finished loading, sampling {per_shard_sample_size} from {shard_size} for training...")
                random_samples = np.random.choice(np.arange(shard_size), size=[min(per_shard_sample_size, shard_size)], replace=False)
                sampled_embs = embeddings[random_samples]
                all_sampled_embs.extend(sampled_embs)
            all_sampled_embs = np.stack(all_sampled_embs).astype(np.float32)
        else:
            print(f"Loading sampled embeddings from {self.sample_path}...")
            with open(self.sample_path, "rb") as fin:
                start_time = time.time()
                all_sampled_embs = pickle.load(fin)
                end_time = time.time()
            print(f"Finished loading sampled embeddings in {end_time - start_time:.2f} seconds.")
        print(f"Sampled {len(all_sampled_embs)} embeddings for training...")
        
        print ("Training index...")
        start_time = time.time()
        self._train_index(all_sampled_embs, self.trained_index_path)
        print ("Finish training (%ds)" % (time.time()-start_time))

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatIP(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.n_subquantizers,
                                       self.code_size,
                                       faiss.METRIC_INNER_PRODUCT
                                       )
        start_index.nprobe = self.probe
        np.random.seed(self.random_seed)

        start_index.cp.seed = self.random_seed
        start_index.pq.cp.seed = self.random_seed 

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        assert index.is_trained and index.ntotal == 0
        
        start_time = time.time()
        prev_domain = None
        # NOTE: the shard id is a absolute id defined in the name
        for shard_id, embed_path in enumerate(self.embed_paths):
            '''
            filename = os.path.basename(embed_path)
            match = re.search(r"passages(\d+)\.pkl", filename)
            shard_id = int(match.group(1))
            to_add = self.get_embs(shard_id=shard_id).copy()
            '''
            # Save an index when changing domain
            if self.save_intermediate_index:
                domain = embed_path.split("/")[-1].split('--')[0]
                if prev_domain is None:
                    prev_domain = domain
                if prev_domain != domain and "passages" not in domain:
                    print(f"Finish adding {prev_domain}, about to add {domain}, saving index...")
                    faiss.write_index(index, index_path.replace('.faiss', f'_{prev_domain}.faiss'))
                    with open(self.meta_file.replace('.faiss.meta', f'_{prev_domain}.faiss.meta'), 'wb') as fout:
                        np.save(fout, np.array(self.index_id_to_file_id))
                    print ('Adding took {} s'.format(time.time() - start_time))
                prev_domain = domain

            with open(embed_path, "rb") as fin:
                _, to_add = pickle.load(fin)
            index.add(to_add)
            file_ids_to_add = [shard_id] * len(to_add)
            self.index_id_to_file_id.extend(file_ids_to_add)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
            with open(self.meta_file.replace('.faiss.meta', f'_.log'), 'w') as fout:
                fout.write(f"Added {shard_id+1} / {len(self.embed_paths)} shards, ({(time.time()-start_time)/60} min)\n")
        
        faiss.write_index(index, index_path)
        with open(self.meta_file, 'wb') as fout:
            np.save(fout, np.array(self.index_id_to_file_id))
        print ('Adding took {} s'.format(time.time() - start_time))
        return index
    
    def build_passage_pos_id_array(self, ):
        convert_pkl_to_jsonl(self.passage_dir)
        passage_pos_ids, passage_filenames = get_passage_pos_ids(self.passage_dir, self.pos_array_save_path, 
                                                                 self.passage_filenames_save_path, self.deprioritized_domains)
        return passage_pos_ids, passage_filenames

    def load_psg_pos_id_array(self,):
        if os.path.exists(self.pos_array_save_path) and os.path.exists(self.passage_filenames_save_path):
            with open(self.pos_array_save_path, 'rb') as f:
                psg_pos_id_array = np.load(f)
            with open(self.passage_filenames_save_path, 'rb') as f:
                passage_filenames = np.load(f, allow_pickle=True)
        else:
            psg_pos_id_array, passage_filenames = self.build_passage_pos_id_array()
        return psg_pos_id_array, passage_filenames

    def _get_passage(self, index_id):
        filename = self.passage_filenames[self.index_id_to_file_id[index_id]]
        position = self.psg_pos_id_array[index_id]
        with open(os.path.join(self.passage_dir, filename), 'r') as file:
            file.seek(position)
            line = file.readline()
        return json.loads(line)

    def _get_domain(self, index_id):
        filename = self.passage_filenames[self.index_id_to_file_id[index_id]]
        return os.path.basename(filename).split("raw_passages")[0].split("--")[0]

    def get_retrieved_passages(self, all_indices):
        domains, passages, db_ids = [], [], []
        for query_indices in all_indices:
            domain_per_query = [self._get_domain(int(index_id)) for index_id in query_indices]
            passages_per_query = [self._get_passage(int(index_id))["text"] for index_id in query_indices]
            db_ids_per_query = [int(index_id) for index_id in query_indices]
            domains.append(domain_per_query)
            passages.append(passages_per_query)
            db_ids.append(db_ids_per_query)
        return domains, passages, db_ids
    
    def search(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        all_domains, all_passages, db_ids = self.get_retrieved_passages(all_indices)
        return all_scores.tolist(), all_domains, all_passages, db_ids
    

    def add_new_embeddings(self, new_embed_paths: List[str], new_passage_dir: str = None):
        """
        Incrementally adds new embeddings and their passage map entries into the current index.
        new_passage_dir should point to the directory containing the new passage .jsonl files.
        """
        assert hasattr(self, 'index') and self.index.is_trained, "Index must be loaded and trained before adding embeddings."
        start_shard_id = max([sid for sid, _ in self.index_id_to_db_id], default=-1) + 1
        start_time = time.time()

        for shard_offset, embed_path in enumerate(tqdm(new_embed_paths)):
            shard_id = start_shard_id + shard_offset
            print(f"Adding embeddings from {embed_path} as shard {shard_id}...")

            with open(embed_path, "rb") as fin:
                _, new_embeds = pickle.load(fin)

            new_embeds = np.array(new_embeds).astype(np.float32)
            self.index.add(new_embeds)

            new_ids = [[shard_id, i] for i in range(len(new_embeds))]
            self.index_id_to_db_id.extend(new_ids)

        # Update index and metadata
        print("Writing updated index and metadata...")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_file, 'wb') as fout:
            pickle.dump(self.index_id_to_db_id, fout)

        # Update passage position map
        if new_passage_dir is not None:
            print(f"Updating passage position map from: {new_passage_dir}")
            print("Building new position map...")
            new_psg_pos_id_map = get_passage_pos_ids(new_passage_dir, None)  # Don't overwrite existing file yet
            print("Merging new position map with existing one...")
            for shard_id, chunk_map in tqdm(new_psg_pos_id_map.items()):
                new_global_shard_id = start_shard_id + shard_id
                self.psg_pos_id_map[new_global_shard_id] = chunk_map

            if self.pos_map_save_path is not None:
                with open(self.pos_map_save_path, 'wb') as f:
                    pickle.dump(self.psg_pos_id_map, f)
                print(f"Updated position map saved to {self.pos_map_save_path}")

        print(f"Added {len(new_embed_paths)} new shards in {(time.time() - start_time)/60:.2f} min.")
