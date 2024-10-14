import datasets
from huggingface_hub import HfApi
import os

base_name = "PKU-Alignment/PKU-SafeRLHF"
new_name = "neeljaycs/pku-rlhf-safe"

def transform(sample):
    prompt = f"BEGINNING OF CONVERSATION: USER: {sample['prompt']} ASSISTANT:"
    
    if sample['safer_response_id'] == 0:
        chosen = sample['response_0']
        rejected = sample['response_1']
    else:
        chosen = sample['response_1']
        rejected = sample['response_0']
    
    return {
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected
    }

def process_split(dataset):
    transform_dataset = dataset.map(transform)
    columns_to_remove = [c for c in transform_dataset.column_names if c not in ['prompt', 'chosen', 'rejected']]
    transform_dataset = transform_dataset.remove_columns(columns_to_remove)
    
    return transform_dataset



def main():
    base_dataset = datasets.load_dataset(base_name)
    transform_dataset = datasets.DatasetDict({
        split: process_split(base_dataset[split]) for split in base_dataset.keys()
    })

    # push to hf
    api = HfApi()
    api.create_repo(repo_id=new_name)
    transform_dataset.push_to_hub(new_name)
    
if __name__ == "__main__":
    main()
    