#!/bin/bash
datastores=("high-quality_cc" "wikipedia_dpr" "pubmed" "arxiv" "github" "stackexchange" "rpj_wikipedia" "math" "reddit")


# Check if an argument is passed
if [ $# -eq 0 ]; then
  echo "Usage: $0 <argument>"
  exit 1
fi

# Get the argument
download_dir=$1
output_dir=$2

# Process the argument
for datastore in "${datastores[@]}"; do
  echo "Building vectors for: $datastores"
  python -m src.main_ric --config-name $datastore tasks.datastore.embedding=true datastore.raw_data_path=$download_dir/$datastore datastore.embedding.output_dir=$output_dir/$datastore
done


