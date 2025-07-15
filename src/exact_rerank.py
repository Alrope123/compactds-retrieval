import os
import json
import pickle as pkl
import logging
from tqdm import tqdm
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import contriever.src.contriever
import contriever.src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'


def embed_queries(args, queries, model, tokenizer, model_name_or_path, cached_embeddings={}):
    print(f"Embedding {len(queries)} text after caching...")
    queries = [q for q in queries if q not in cached_embeddings]

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
            for k, q in enumerate(tqdm(queries)):
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

            for k, q in enumerate(tqdm(queries)):
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
    for query, embedding in zip(queries, embeddings):
        cached_embeddings[query] = embedding

    if args.get('update_exact_embedding_cache', True):
        with open(args.get("exact_embedding_cache_path", "cache.pkl"), 'wb') as fout:
            pkl.dump(cached_embeddings, fout)

    return cached_embeddings


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



def exact_rerank_topk(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation
    ds_domain = cfg.datastore.domain
   
    output_path = get_search_output_path(cfg)
    assert os.path.exists(output_path), f"Search output path does not exist: {output_path}. Please run search first."
        
    # load model and evaluation data
    assert "exact_encoder" in cfg.model, "Please specify the exact encoder model in the config file."
    assert "exact_tokenizer" in cfg.model, "Please specify the exact tokenizer model in the config file."
    logging.info(f"Loading model from: {cfg.model.exact_encoder}")
    model_name_or_path = cfg.model.exact_encoder
    tokenizer_name_or_path = cfg.model.exact_tokenizer
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
    
    # load retrieved result data
    results = []
    with open(output_path, 'r') as fin:
        for line in fin:
            ex = json.loads(line)
            results.append(ex)
    
    logging.info(f"Doing exact reranking for {len(results)} total evaluation samples...")
    if os.path.exists(eval_args.search.get('exact_embedding_cache_path', "cache.pkl")):
        logging.info(f"Loading cached exact embeddings from {eval_args.search.get('exact_embedding_cache_path', 'cache.pkl')}")
        with open(eval_args.search.get('exact_embedding_cache_path', "cache.pkl"), 'rb') as fin:
            cached_embeddings = pkl.load(fin)
    else:
        if not os.path.exists(os.path.dirname(eval_args.search.get('exact_embedding_cache_path', "cache.pkl"))) and \
           os.path.dirname(eval_args.search.get('exact_embedding_cache_path', "cache.pkl")) != "":
            logging.info(f"Creating directory for exact embedding cache: {os.path.dirname(eval_args.search.get('exact_embedding_cache_path', 'cache.pkl'))}")
            os.makedirs(os.path.dirname(eval_args.search.get('exact_embedding_cache_path', "cache.pkl")), exist_ok=True)
        cached_embeddings = {}

    # Get all needed embeddings
    queries = []
    ctxs = []
    for res in results:
        queries.append(res['raw_query']) 
        ctxs.extend([ctx['retrieval text'] for ctx in res['ctxs'] if ctx is not None])
    
    text_to_embeddings = embed_queries(eval_args.search, queries + ctxs, query_encoder, query_tokenizer, model_name_or_path, 
                                        cached_embeddings)
    
    print("Calculating the exact similarity scores...")
    for res in results:
        query_embedding = text_to_embeddings[res['raw_query']].reshape(1, -1)
        ctxs_embedding = np.vstack([text_to_embeddings[ctx['retrieval text']] for ctx in res['ctxs']])
        similarities = cosine_similarity(query_embedding, ctxs_embedding)
        for ctx, similarity in zip(res['ctxs'], similarities[0]):
            ctx['grit score'] = float(similarity)
    
    base_name, ext = os.path.splitext(output_path) 
    new_output_path = base_name + "_exact_searched" + ext
    safe_write_jsonl(results, new_output_path)
    


def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))

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
    exact_rerank_topk(cfg)
