import os
import pickle
import json
from tqdm import tqdm
import re
import glob
import numpy as np

def get_index_dir_and_embedding_paths(cfg, index_shard_ids=None, deprioritized_domains=[]):
    embedding_args = cfg.datastore.embedding
    index_args = cfg.datastore.index
    index_type = cfg.datastore.index.index_type

    # index passages
    index_shard_ids = index_shard_ids if index_shard_ids is not None else index_args.get('index_shard_ids', None)
    
    if index_shard_ids:
        index_shard_ids = [int(i) for i in sorted(index_shard_ids)]
        embedding_paths = [os.path.join(embedding_args.embedding_dir, embedding_args.prefix + f"_{shard_id:02d}.pkl")
                       for shard_id in index_shard_ids]

        # name the index dir with all shard ids included in this index, i.e., one index for multiple passage shards
        index_dir_name = '_'.join([str(shard_id) for shard_id in sorted(index_shard_ids)])
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index_{index_type}/{index_dir_name}')
        
    else:
        embedding_paths = glob.glob(index_args.passages_embeddings)
        print(f"passages_embeddings: {index_args.passages_embeddings}")
        # put some domains to the back
        # ["massiveds-rpj_arxiv", "massiveds-rpj_github", "massiveds-rpj_book", "lb_full"]
        deprioritized_domains_index = {domain: i+1 for i, domain in enumerate(deprioritized_domains)}
        def sort_func(x):
            domain = x.split("/")[-1].split(f'{embedding_args.prefix}')[0].split('--')[0] 
            rank, shard_idx = x.split("/")[-1].split(f'{embedding_args.prefix}')[-1].split(".pkl")[0].split("_")
            if domain not in deprioritized_domains_index:
                depriortized = 0
            else:
                depriortized = deprioritized_domains_index[domain]
            return depriortized, domain, int(rank), int(shard_idx)
        embedding_paths = sorted(embedding_paths, key=sort_func)
        print("DEBUG: sorted embedding paths:")
        print("\n".join(embedding_paths))
        embedding_paths = embedding_paths if index_args.num_subsampled_embedding_files == -1 else embedding_paths[0:index_args.num_subsampled_embedding_files]
        
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index_{index_type}')
    
    return index_dir, embedding_paths


def convert_pkl_to_jsonl(passage_dir):
    if os.path.isdir(passage_dir):
        filenames = os.listdir(passage_dir)
        pkl_files = [filename for filename in filenames if '.pkl' in filename]
        jsonl_files = [filename for filename in filenames if '.jsonl' in filename]
        print (f"Found {len(pkl_files)} pkl files and {len(jsonl_files)} jsonl files under {passage_dir}")
        if len(pkl_files)<=len(jsonl_files):
            return
        print(f"Converting passages to JSONL data format: {passage_dir}")
    elif os.path.isfile(passage_dir):
        assert '.pkl' in passage_dir or '.jsonl' in passage_dir, f"File {passage_dir} is neither a .pkl nor a .jsonl file."
        if '.pkl' in passage_dir:
            pkl_files = [passage_dir]
        else:
            return
    else:
        print(f"{passage_dir} does not exist or is neither a file nor a directory.")
        raise AssertionError
        
    for file in tqdm(pkl_files):
        
        # Create the JSONL file name
        file_path = os.path.join(passage_dir, file)
        jsonl_file = file_path.replace('.pkl', '.jsonl')
        
        if os.path.exists(jsonl_file):
            continue
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Length of {file}: {len(data)}")

        # Save the data to the JSONL file
        with open(jsonl_file, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')
    print("All pickle files have been converted to JSONL files.")

def get_passage_pos_ids(passage_dir, pos_array_save_path, filenames_save_path, deprioritized_domains=[]):
    if os.path.isdir(passage_dir):
        print(f"Generating id2pos for {passage_dir}")
        filenames = os.listdir(passage_dir)
        jsonl_files = [filename for filename in filenames if '.jsonl' in filename]
        
        deprioritized_domains_index = {domain: i+1 for i, domain in enumerate(deprioritized_domains)}
        def sort_func(x):
            domain = x.split("/")[-1].split(f'raw_passages')[0].split('--')[0] 
            rank = int(x.split('passages_')[-1].split("-")[0])
            shard_idx = int(x.split("-of-")[0].split("-")[-1])
            if domain not in deprioritized_domains_index:
                depriortized = 0
            else:
                depriortized = deprioritized_domains_index[domain]
            return depriortized, domain, int(rank), int(shard_idx)
        jsonl_files = sorted(
            jsonl_files,
            key=sort_func)
        print("DEBUG: Sorted JSONL files:")
        print("\n".join(jsonl_files))

        pos_id_array = []
        file_names = []
        total = 0
        for shard_id, filename in enumerate(tqdm(jsonl_files)):
            file_names.append(filename)
            file_path = os.path.join(passage_dir, filename)
            
            with open(file_path, 'r') as file:
                position = file.tell()
                line = file.readline()
                doc_id = 0
                while line:
                    pos_id_array.append(position)
                    doc_id += 1
                    position = file.tell()
                    line = file.readline()
            total += doc_id - 1

    elif os.path.isfile(passage_dir):
        # NOTE: deprecated feature, will be removed in future release.
        file_path = passage_dir.replace('.pkl', '.jsonl')
        print(f"Generating id2pos for {file_path}")

        pos_id_array = []
        file_names = []
        
        file_names.append(file_path)
        with open(file_path, 'r') as file:
            position = file.tell()
            line = file.readline()
            doc_id = 0
            while line:
                pos_id_array.append(position)
                doc_id += 1
                position = file.tell()
                line = file.readline()
    else:
        print(f"{passage_dir} does not exist or is neither a file nor a directory.")
        raise AssertionError
    
    # Save the output array to a pickle file
    if pos_array_save_path is not None:
        with open(pos_array_save_path, 'wb') as f:
            np.save(f, np.array(pos_id_array, dtype=np.int64))
        print(f"Output array saved to {pos_array_save_path}")

    # Save the output filenames to a pickle file
    if filenames_save_path is not None:
        with open(filenames_save_path, 'wb') as f:
            np.save(f, np.array(file_names, dtype=object))
        print(f"Output filenames saved to {filenames_save_path}")

    return pos_id_array, file_names


if __name__ == '__main__':
    passage_dir = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards'
    convert_pkl_to_jsonl(passage_dir)
    
    pos_map_save_path = '/gscratch/zlab/rulins/data/scaling_out/passages/pes2o_v3/16-shards/pos_map.pkl'
    get_passage_pos_ids(passage_dir, pos_map_save_path)
