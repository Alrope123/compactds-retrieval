import os
import re
import json
import time
import pickle
import faiss
import numpy as np
import torch

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class FlatIndexer(object):

    def __init__(self, 
                embed_paths=None,
                index_path=None,
                meta_file=None,
                passage_dir=None,
                deprioritized_domains=[],
                pos_array_save_path=None,
                passage_filenames_save_path=None,
                dimension=768,
                save_intermediate_index=False,
                ):
    
        self.embed_paths = embed_paths
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.passage_dir = passage_dir
        self.deprioritized_domains = deprioritized_domains
        self.pos_array_save_path = pos_array_save_path
        self.dimension=dimension
        self.save_intermediate_index = save_intermediate_index
        self.cuda = False

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_file_id = self.load_index_id_to_file_id()
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.index_id_to_file_id = []
            print ("Building index...")
            self._build_index()
        
        if self.pos_array_save_path is not None:
            self.psg_pos_id_array, self.passage_filenames = self.load_psg_pos_id_array()

    def _build_index(self,):
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
                    faiss.write_index(self.index, self.index_path.replace('.faiss', f'_{prev_domain}.faiss'))
                    with open(self.meta_file.replace('.faiss.meta', f'_{prev_domain}.faiss.meta'), 'wb') as fout:
                        np.save(fout, np.array(self.index_id_to_file_id))
                    print ('Adding took {} s'.format(time.time() - start_time))
                prev_domain = domain

            with open(embed_path, "rb") as fin:
                _, to_add = pickle.load(fin)
            self.index.add(to_add)
            file_ids_to_add = [shard_id] * len(to_add)
            self.index_id_to_file_id.extend(file_ids_to_add)
            print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
            with open(self.meta_file.replace('.faiss.meta', f'_.log'), 'w') as fout:
                fout.write(f"Added {shard_id+1} / {len(self.embed_paths)} shards, ({(time.time()-start_time)/60} min)\n")
        
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_file, 'wb') as fout:
            np.save(fout, np.array(self.index_id_to_file_id))
        print ('Adding took {} s'.format(time.time() - start_time))
        
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
