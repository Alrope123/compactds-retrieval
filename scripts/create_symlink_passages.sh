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
mkdir $output_dir/passages

# Create symlinks to the vector files
for datastore in "${datastores[@]}"; 
do
  datastore_path="$input_dir/$datastore/passages/$datastore"
  find $path -name "*.pkl" | while read file; do
    ln -s "$file" $output_dir/passages/$(basename "$(dirname "$file")")--$(basename "$file")
  done
done
