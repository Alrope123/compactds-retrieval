#!/bin/bash

# Check if an argument is passed
if [ $# -eq 0 ]; then
  echo "Usage: $0 <argument>"
  exit 1
fi

output_dir=$1

# Download the sharded index files
python scripts/download_index.py --output_path output_dir

# Combine the shards
cat output_dir/embeddings/index_IVFPQ/index_IVFPQ.100000000.768.65536.64.faiss* > index_IVFPQ.100000000.768.65536.64.faiss

# Remove shard files
rm output_dir/embeddings/index_IVFPQ/index_IVFPQ.100000000.768.65536.64.faiss_*