import os
import re

import argparse
import csv
import logging
import pickle
import pdb
from tqdm import tqdm

import numpy as np
import torch
import math

import boto3
import smart_open
import json
import gzip

from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import contriever.src.contriever
import contriever.src.normalize_text

from src.data import fast_load_jsonl_shard_full_files

device = 'cuda' if torch.cuda.is_available()  else 'cpu'

def is_sentence_transformers(model_name_or_path):
    return "sentence-transformers" in model_name_or_path or \
            "intfloat" in model_name_or_path or \
            "Snowflake" in model_name_or_path or \
            "GritLM" in model_name_or_path

def embed_passages(args, passages, model, tokenizer, shard_id, num_shards):
    if is_sentence_transformers(args.model_name_or_path):
        allids, alltext = [], []
        for k, p in tqdm(enumerate(passages)):
            allids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = contriever.src.normalize_text.normalize(text)
            alltext.append(text)
        
        with torch.no_grad():
            if "GritLM" in args.model_name_or_path:
                allembeddings = model.encode(alltext, batch_size=64, instruction="<|embed|>\n")
            else:
                allembeddings = model.encode(alltext, batch_size=64)  # default is 512, but got oom
    
    elif "meta-llama" in args.model_name_or_path:
        total = 0
        allids, allembeddings = [], []
        batch_ids, batch_text = [], []
        tot_psgs = len(passages)

        with torch.no_grad():
            for k, p in tqdm(enumerate(passages)):
                batch_ids.append(p["id"])
                
                # Prepare text for encoding
                if args.no_title or "title" not in p:
                    text = p["text"]
                else:
                    text = p["title"] + " " + p["text"]
                if args.lowercase:
                    text = text.lower()
                if args.normalize_text:
                    text = contriever.src.normalize_text.normalize(text)
                batch_text.append(text)

                if len(batch_text) == args.per_gpu_batch_size or k == tot_psgs - 1:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        max_length=args.passage_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    print(f"EMBEDDING UP TO PSG {k} out of {tot_psgs} (in shard {shard_id} of {num_shards})")
                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)  # Get model output

                    if "contriever" not in args.model_name_or_path:
                        hidden_states = output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
                        attention_mask = encoded_batch["attention_mask"]  # Shape: (batch_size, seq_len)

                        seq_len = hidden_states.shape[1]  # Get sequence length (L)
                        indices = torch.arange(1, seq_len + 1, dtype=torch.float32, device=hidden_states.device)  # Token positions

                        # Zero out weights for padding tokens
                        indices = indices * attention_mask  # Multiply by mask to remove padding influence
                        weight_sum = torch.sum(indices, dim=1, keepdim=True)  # Sum of non-padding weights per passage
                        
                        # Avoid division by zero (handle all-padding cases)
                        weight_sum = torch.where(weight_sum == 0, torch.tensor(1.0, device=hidden_states.device), weight_sum)

                        # Compute normalized weights
                        weights = indices / weight_sum  # Normalize weights
                        weights = weights.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1] for broadcasting

                        # Compute final weighted embedding
                        weighted_embedding = torch.sum(weights * hidden_states, dim=1)  # Weighted sum over tokens

                        embeddings = weighted_embedding.cpu()

                    total += len(batch_ids)
                    allids.extend(batch_ids)
                    allembeddings.append(embeddings)

                    batch_text = []
                    batch_ids = []

                    if k % 10000 == 0 and k > 0:
                        print(f"Encoded passages {total}")

        allembeddings = torch.cat(allembeddings, dim=0).numpy()

    else:
        total = 0
        allids, allembeddings = [], []
        batch_ids, batch_text = [], []
        tot_psgs = len(passages)
        with torch.no_grad():
            for k, p in tqdm(enumerate(passages)):
                batch_ids.append(p["id"])
                if args.no_title or not "title" in p:
                    text = p["text"]
                else:
                    text = p["title"] + " " + p["text"]
                if args.lowercase:
                    text = text.lower()
                if args.normalize_text:
                    text = contriever.src.normalize_text.normalize(text)
                batch_text.append(text)

                if len(batch_text) == args.per_gpu_batch_size or k == tot_psgs - 1:

                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        max_length=args.passage_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    print(f"EMBEDDING UP TO PSG {k} out of {tot_psgs} (in shard {shard_id} of {num_shards})")
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    embeddings = model(**encoded_batch)  # shape: (per_gpu_batch_size, hidden_size)
                    if "contriever" not in args.model_name_or_path:
                        # assume in hf form
                        embeddings = embeddings.last_hidden_state[:, 0, :]

                    embeddings = embeddings.cpu()
                    
                    total += len(batch_ids)
                    allids.extend(batch_ids)
                    allembeddings.append(embeddings)

                    batch_text = []
                    batch_ids = []
                    if k % 10000 == 0 and k > 0:
                        print(f"Encoded passages {total}")
        
        allembeddings = torch.cat(allembeddings, dim=0).numpy()

    allembeddings = allembeddings.astype(np.float16)
    return allids, allembeddings


def get_sharded_passages(args, all_passages):
    total_num_passages = len(all_passages)
    shard_size = total_num_passages // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = total_num_passages
    
    passages = all_passages[start_idx: end_idx]
    print(f"Using {len(passages)} passages from idx {start_idx} to {end_idx}.")
    return passages

def get_shard_specs(args, file_paths, file_sizes):
    total_size = sum(file_sizes)

    if args.get("max_shard_size",None):
        shard_size = args.max_shard_size
        num_shards = math.floor(total_size/shard_size) + 1
    elif args.get("num_shards",None):
        shard_size = total_size / args.num_shards
        num_shards = args.num_shards

    return num_shards,shard_size

def get_file_paths_and_sizes(args):
    file_sizes = []
    if "s3://" in args.raw_data_path and (
            args.raw_data_path.endswith(".jsonl") or args.raw_data_path.endswith(".gz")):
        file_paths = [args.raw_data_path]
    elif "s3://" in args.raw_data_path:
        from src.utils_paths import glob_path
        file_paths = sorted(
                list(glob_path(os.path.join(args.raw_data_path, "*.gz")))
        )
    else:
        file_paths = []
        for file in os.listdir(args.raw_data_path):
            file_paths.append(os.path.join(args.raw_data_path, file))
            # file_sizes.append(os.path.getsize(file_path))
        file_paths = sorted(file_paths)

    return file_paths,file_sizes

def get_file_partitions(args):
    file_paths,_ = get_file_paths_and_sizes(args)

    rank = int(os.environ.get("BEAKER_REPLICA_RANK"))
    world_size = int(os.environ.get("BEAKER_REPLICA_COUNT"))
    
    print (f"Distributing {len(file_paths)} files into world_size={world_size}")

    # Distribute files across processes
    files_per_process = len(file_paths) / world_size
    start_idx = int(rank * files_per_process)
    end_idx = int((rank + 1) * files_per_process) if rank < world_size - 1 else len(file_paths)
    partition_file_paths = file_paths[start_idx:end_idx]
    # partition_file_sizes = file_sizes[start_idx:end_idx]

    print(f"This worker (rank {rank}) handling files:\n {partition_file_paths[0]}\n to\n {partition_file_paths[-1]}")

    return partition_file_paths,rank


def generate_passage_embeddings(cfg):
    if cfg.model.get("sparse_retriever", None):
        print(f"No need to run the embedding step for sparse retrieval, skipping...")

    else:
        args = cfg.datastore.embedding
        
        logging.info(f"Loading retriever model from {args.model_name_or_path}...")
        if "meta-llama" in args.model_name_or_path:
            tokenizer_name_or_path = args.tokenizer if args.get('tokenizer', None) else args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(args.model_name_or_path)
        elif "GritLM" in args.model_name_or_path:
            from gritlm import GritLM
            tokenizer  = None
            model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
        elif "contriever" in args.model_name_or_path:
            model, tokenizer, _ = contriever.src.contriever.load_retriever(args.model_name_or_path)
        elif "dragon" in args.model_name_or_path:
            tokenizer_name_or_path = args.tokenizer if args.get('tokenizer', None) else args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            model = AutoModel.from_pretrained(args.model_name_or_path)
        elif is_sentence_transformers(args.model_name_or_path):
            tokenizer = None
            model = SentenceTransformer(args.model_name_or_path)
        else:
            print(f"{args.model_name_or_path} is not supported!")
            raise AttributeError
        
        # arg_num_shards = args.get("num_shards",None)
        # arg_max_shard_size = args.get("max_shard_size",None)
        # assert sum([bool(arg_num_shards),bool(arg_max_shard_size)]) == 1 , "Specify either datastore.embedding.num_shards or datastore.embedding.max_shard_size, but not both"
        
        model.eval()
        model = model.cuda()
        if not args.no_fp16:
            model = model.half()
        

        partition_file_paths,rank = get_file_partitions(args)
        # num_shards,shard_size = get_shard_specs(args, partition_file_paths,partition_file_sizes)


        num_files = args.max_files_per_shard if args.get("max_files_per_shard",None) else len(partition_file_paths)
        start_list = range(0,len(partition_file_paths),num_files)
        num_shards = len(start_list)

        for shard_id, shard_start in enumerate(start_list):
            print(f"Processing EMBEDDING SHARD {shard_id} out of {num_shards} shards (for worker {rank})")
            embedding_shard_save_path = os.path.join(args.embedding_dir, args.prefix + f"{rank}_{shard_id:02d}.pkl")
            
            if os.path.exists(embedding_shard_save_path) and args.get("use_saved_if_exists", "true"):
                print(f"Embeddings exist in {embedding_shard_save_path}")
                continue
            
            shard_passages = fast_load_jsonl_shard_full_files(args, partition_file_paths, rank, shard_id, shard_start, num_files, num_shards)
            if args.get("logloc",None):
                logpath = Path(args.logloc)
                logpath.mkdir(parents=True, exist_ok=True)
                with open(os.path.join(args.logloc,f"{rank}_{shard_id:02d}.json"),"w") as logout:
                    logout.write(json.dumps(shard_passages,indent=4))
            #shard_passages = fast_load_jsonl_shard(args, shard_id) #, return_all_passages=True)
            
            if len(shard_passages) == 0:
                allids = []
                allembeddings = np.empty((0, cfg.datastore.index.projection_size), dtype=np.float16)
            else:
                allids, allembeddings = embed_passages(args, shard_passages, model, tokenizer, shard_id, num_shards)
            os.makedirs(args.embedding_dir, exist_ok=True)
            print(f"Saving {len(allids)} passage embeddings to {embedding_shard_save_path}.")
            with smart_open.open(embedding_shard_save_path, mode="wb") as file:
                pickle.dump((allids, allembeddings), file)

            print(f"Processed {len(allids)} passages in the {shard_id}-th (out of {num_shards}) shard.\nWritten to {embedding_shard_save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_data_path", type=str, default=None, help="Path to passages (.jsonl or .tsv file)")
    parser.add_argument("--embedding_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="Maximum number of words in a passage, the length will be further cut by passage_maxlength")
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    args = parser.parse_args()

    generate_passage_embeddings(args)
