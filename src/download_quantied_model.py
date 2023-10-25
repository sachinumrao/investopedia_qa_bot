import os

from huggingface_hub import hf_hub_download


def download_model_from_hub(model_save_path, model_repo, model_file):
    model_path = os.path.abspath(model_save_path)
    print("Model Path: ", model_path)
    print("Model Repo: ", model_repo)
    print("Model File: ", model_file)

    return hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=model_path,
        local_dir_use_symlinks=True,
    )


if __name__ == "__main__":
    model_repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_save_path = "./../models/"
    download_model_from_hub(model_save_path, model_repo, model_file)
