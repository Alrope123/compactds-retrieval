import os
import json
import pickle as pkl
import logging
import random
from tqdm import tqdm
import re
from omegaconf import ListConfig

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
try:
    from pyserini.search.lucene import LuceneSearcher
except:
    logging.warning("Failed to import pyserini! Please install it from https://github.com/castorini/pyserini/tree/master.")

import contriever.src.contriever
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import load_eval_data
from src.index import Indexer, get_bm25_index_dir

os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'


def embed_queries(args, queries, model, tokenizer, model_name_or_path):
    if "GritLM" in model_name_or_path:
        all_question = []
        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = contriever.src.normalize_text.normalize(q)
            all_question.append(q)

        embeddings = model.encode(all_question, batch_size=min(128, args.per_gpu_batch_size), instruction="<|embed|>\n")  # sentence-transformer has extra memory overhead and can only support a smaller batch size
    elif "sentence-transformers" in model_name_or_path:
        all_question = []
        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = contriever.src.normalize_text.normalize(q)
            all_question.append(q)
        
        embeddings = model.encode(all_question, batch_size=min(128, args.per_gpu_batch_size))  # sentence-transformer has extra memory overhead and can only support a smaller batch size
    elif "meta-llama" in model_name_or_path:
        model.eval()
        embeddings, batch_question = [], []

        with torch.no_grad():
            for k, q in tqdm(enumerate(queries)):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = contriever.src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)

                    if "contriever" not in model_name_or_path:
                        hidden_states = output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
                        attention_mask = encoded_batch["attention_mask"]  # Shape: [batch_size, seq_len]
                        
                        seq_len = hidden_states.shape[1]  # L (max sequence length)
                        indices = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)  # Position indices

                        # Zero out weights for padding tokens
                        indices = indices * attention_mask  # Zero out padding indices
                        weight_sum = torch.sum(indices, dim=1, keepdim=True)  # Sum of non-padding weights per sequence

                        # Avoid division by zero (if a sequence is entirely padding, set weight_sum to 1 to prevent NaN)
                        weight_sum = torch.where(weight_sum == 0, torch.tensor(1.0, device=device), weight_sum)

                        weights = indices / weight_sum  # Normalize weights
                        weights = weights.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1] for broadcasting

                        # Compute weighted sum, ignoring paddings
                        weighted_embedding = torch.sum(weights * hidden_states, dim=1)  # Shape: [batch_size, hidden_dim]
                        embeddings.append(weighted_embedding.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0).numpy()
    else:
        model.eval()
        embeddings, batch_question = [], []
        with torch.no_grad():

            for k, q in tqdm(enumerate(queries)):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = contriever.src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)
                    if "contriever" not in model_name_or_path:
                        output = output.last_hidden_state[:, 0, :]
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0).numpy()
    
    print(f"Questions embeddings shape: {embeddings.shape}")

    if args.get('cache_query_embedding', False):
        with open(args.query_embedding_save_path, 'wb') as fout:
            pkl.dump(embeddings, fout)

    return embeddings



def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits


def add_passages_to_eval_data(data, domains, passages, scores, db_ids, valid_query_idx, domain=None):
    # add passages to original data
    assert len(valid_query_idx) == len(passages)
    idx = 0
    for i, d in enumerate(data):
        if i in valid_query_idx:
            ex_scores = scores[idx]
            ex_scores = [str(score) for score in scores[idx]]
            ctxs_num = len(passages[0])
            d["ctxs"] = [
                {
                    "id": db_ids[idx][c],
                    "source": domains[idx][c],
                    "retrieval text": passages[idx][c],
                    "retrieval score": ex_scores[c],
                }
                for c in range(ctxs_num)
            ]
            idx += 1
        else:
            d["ctxs"] = [None]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def get_search_output_path(cfg, index_shard_ids=None):
    eval_args = cfg.evaluation
    if index_shard_ids:
        shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        output_dir = os.path.join(eval_args.eval_output_dir, shards_postfix)
    else:
        output_dir = eval_args.eval_output_dir
    
    index_type = cfg.datastore.index.index_type
    if "IVF" in index_type:
        postfix = f"_{index_type}.{cfg.datastore.index.ncentroids}"
        if "PQ" in index_type:
            postfix = f"{postfix}.{cfg.datastore.index.n_subquantizers}"
        postfix = f"{postfix}.{cfg.datastore.index.probe}"
    else:
        postfix = ""

    output_path = os.path.join(output_dir + postfix, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def get_merged_search_output_path(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    merged_postfix = ''
    for index_shard_ids in sorted(index_shard_ids_list, key=lambda x: int(x[0])):
        shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        merged_postfix += '-' + shards_postfix
    merged_postfix = merged_postfix.strip('-')

    output_dir = os.path.join(eval_args.eval_output_dir, merged_postfix)
    output_path = os.path.join(output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def get_merged_subsampled_search_output_path(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    merged_postfix = ''
    for index_shard_ids in sorted(index_shard_ids_list, key=lambda x: int(x[0])):
        shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        merged_postfix += '-' + shards_postfix
    merged_postfix = merged_postfix.strip('-')

    if cfg.evaluation.search.get('topk_subsample_p', None):
        seed = cfg.evaluation.search.get('subsample_seed', 1000)
        output_dir = os.path.join(eval_args.eval_output_dir, os.path.join(f'subsampled_{cfg.evaluation.search.topk_subsample_p}_seed_{seed}', merged_postfix))
    else:
        output_dir = os.path.join(eval_args.eval_output_dir, merged_postfix)

    output_path = os.path.join(output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def search_dense_topk(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation
    ds_domain = cfg.datastore.domain
   
    do_search = True

    if index_args.index_shard_ids:

        if isinstance(index_args.index_shard_ids[0], ListConfig):
            print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
            index_shard_ids_list = index_args.index_shard_ids
        else:
            print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
            index_shard_ids_list = [index_args.index_shard_ids]

        all_exist = True
        for index_shard_ids in index_shard_ids_list:
            # check if all search results exist
            output_path = get_search_output_path(cfg, index_shard_ids)
            all_exist = all_exist and os.path.exists(output_path)
        
        if all_exist and not eval_args.search.overwrite:
            logging.info(f'All search results for {index_args.index_shard_ids} exist, skipping searching.')
            do_search = False

    else:
        output_path = get_search_output_path(cfg)
        do_search = not os.path.exists(output_path)
        
    if True: #do_search:
        # load model and evaluation data
        logging.info(f"Loading model from: {cfg.model.datastore_encoder}")
        model_name_or_path = cfg.model.query_encoder
        tokenizer_name_or_path = cfg.model.query_tokenizer
        if "meta-llama" in model_name_or_path:
            query_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            query_tokenizer.pad_token = query_tokenizer.eos_token
            query_encoder = AutoModel.from_pretrained(model_name_or_path)
        elif "GritLM" in model_name_or_path:
            from gritlm import GritLM
            query_tokenizer  = None
            query_encoder = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
        elif "contriever" in model_name_or_path:
            query_encoder, query_tokenizer, _ = contriever.src.contriever.load_retriever(model_name_or_path)
        elif "dragon" in model_name_or_path:
            query_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            query_encoder = AutoModel.from_pretrained(model_name_or_path)
        elif "sentence-transformers" in model_name_or_path:
            query_tokenizer = None
            query_encoder = SentenceTransformer(model_name_or_path)
        elif "e5" in model_name_or_path:
            query_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            query_encoder = AutoModel.from_pretrained(model_name_or_path)
        else:
            print(f"{model_name_or_path} is not supported!")
            raise AttributeError

        query_encoder.eval()
        query_encoder = query_encoder.to(device)
        if not index_args.no_fp16:
            query_encoder = query_encoder.half()
        
        # load eval data
        data = load_eval_data(cfg)

        queries = []
        valid_query_idx = []
        for idx, ex in enumerate(data):
            raw_query = ex["raw_query"]
            if raw_query:
                queries.append(ex["raw_query"])
                valid_query_idx.append(idx)
        
        logging.info(f"Searching for {len(queries)} queries from {len(data)} total evaluation samples...")
        if eval_args.search.get('cache_query_embedding', False) and os.path.exists(eval_args.search.get('query_embedding_save_path', "")):
            logging.info(f"Loading query embeddings from {eval_args.search.query_embedding_save_path}")
            with open(eval_args.search.query_embedding_save_path, 'rb') as fin:
                questions_embedding = pkl.load(fin)
        else:
            questions_embedding = embed_queries(eval_args.search, queries, query_encoder, query_tokenizer, model_name_or_path)
        if eval_args.search.get('cache_query_embedding_only', False):
            return

        output_path = get_search_output_path(cfg)

        # TODO: load index and perform search
        logging.info("Loading or constructing the datastore...")
        index = Indexer(cfg)

        logging.info("Searching for the queries...")
        all_scores, all_domains, all_passages, db_ids = index.search(questions_embedding, eval_args.search.n_docs)
        
        # todo: double check valid_query_idx
        logging.info(f"Adding documents to eval data...")
        add_passages_to_eval_data(data, all_domains, all_passages, all_scores, db_ids, valid_query_idx, domain=ds_domain)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safe_write_jsonl(data, output_path)

def subsample_by_coin_flip(items, probability):
    subsampled_list = []
    for item in items:
        # Perform a coin flip with probability p of being True (keep the item)
        if random.random() < probability:
            subsampled_list.append(item)
    return subsampled_list




def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))


def search_sparse_topk(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building a BM25 index over {len(index_args.index_shard_ids)} shards...")
        index_shard_ids_list = [i for index_shards in index_args.index_shard_ids for i in index_shards]
    else:
        print(f"Single-index mode: building a BM25 index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = index_args.index_shard_ids

    # check if all search results exist
    output_path = get_search_output_path(cfg, index_shard_ids_list)
    all_exist = os.path.exists(output_path)

    if all_exist and not eval_args.search.overwrite:
        logging.info(f'All search results for {index_args.index_shard_ids} exist, skipping searching.')
    
    else:
        # load eval data
        data = load_eval_data(cfg)
        logging.info(f"Searching for {len(data)} total evaluation samples...")

        # load index
        bm25_index_path = os.path.join(get_bm25_index_dir(cfg, index_shard_ids_list), 'index')
        assert os.path.exists(bm25_index_path), f"The index path does not exist, please build the index first\nMissing: {bm25_index_path}"
        logging.info(f"Loading BM25 index from {bm25_index_path}")
        searcher = LuceneSearcher(bm25_index_path)

        for ex in tqdm(data):
            query = ex["raw_query"]
            if query:
                hits = searcher.search(query, cfg.evaluation.search.n_docs)
                ex["ctxs"] = [
                    {
                        # "id": int(ex["id"]),
                        "retrieval text": json.loads(searcher.doc(hits[i].docid).raw())["contents"],
                        "retrieval score": hits[i].score,
                        } for i in range(len(hits))
                ]
            else:
                ex["ctxs"] = [None]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safe_write_jsonl(data, output_path)


def safe_write_jsonl(data, output_file):
    success = False
    try:
        with open(output_file, 'w') as fout:
            for ex in data:
                fout.write(json.dumps(ex) + "\n")
            success = True
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
    # If an error was raised, and success is still False, delete the file
        if not success and os.path.exists(output_file):
            os.remove(output_file)
            print(f"File '{output_file}' has been deleted due to an error.")


def search_topk(cfg):
    if cfg.model.get("sparse_retriever", None):
        search_sparse_topk(cfg)
    else:
        search_dense_topk(cfg)
