import argparse
import os
from huggingface_hub import HfApi


def main(args):
    api = HfApi()
    os.makedirs(args.output_path, exist_ok=True)
    api.dataset_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        local_dir=args.output_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the Huggingface dataset to download", default="alrope/CompactDS-102GB-queries")
    parser.add_argument("--output_path", type=str, help="Path to the local directory to save the downloaded files")
    args = parser.parse_args()
    main(args)
