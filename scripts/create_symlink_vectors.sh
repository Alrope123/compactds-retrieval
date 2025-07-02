#!/bin/bash
datastores=("high-quality_cc" "wikipedia_dpr" "pubmed" "arxiv" "github" "stackexchange" "wikipedia_rpj" "math" "reddit" "pes2o")


# Check if an argument is passed
if [ $# -eq 0 ]; then
  echo "Usage: $0 <argument>"
  exit 1
fi

# Get the argument
input_dir=$1
output_dir=$2

# Create directories to contain the aggregated vectors 
mkdir $output_dir/embeddings

# Create symlinks to the vector files
for datastore in "${datastores[@]}"; 
do
  datastore_path="$input_dir/$datastore/embeddings/facebook/contriever-msmarco/$datastore"
  find $datastore_path -name "*passages*.pkl" | while read file; do
    ln -s "$file" $output_dir/embeddings/$(basename "$(dirname "$file")")--$(basename "$file")
  done
done


