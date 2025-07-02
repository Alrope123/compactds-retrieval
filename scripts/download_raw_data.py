import argparse
import os
from huggingface_hub import HfApi


def main(args):
    api = HfApi()
    os.makedirs(args.output_path, exist_ok=True)
    if args.subfolder_path is None:
        api.dataset_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            local_dir=args.output_path,
        )
    else:
        api.snapshot_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            local_dir=args.output_path,
            allow_patterns=f"{args.subfolder_path}/*",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the Huggingface dataset to download", default="alrope/CompactDS-102GB-raw-text")
    parser.add_argument("--output_path", type=str, help="Path to the local directory to save the downloaded files")
    parser.add_argument("--subfolder_path", type=str, default=None, help="Path to the subfolder in the HF repo to download")
    args = parser.parse_args()
    main(args)
