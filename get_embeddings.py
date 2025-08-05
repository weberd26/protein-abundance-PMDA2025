import argparse
import torch
import numpy as np
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_fasta_file(file_path, sample_fraction=1, seed =42): # REMEMBER to choose the sample fraction of interest
    """Load 10% of sequences and IDs from a FASTA file."""
    data_seqs = []
    data_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                data_ids.append(line.split('>')[1].split('\n')[0])
            else:
                data_seqs.append(line.split('\n')[0])
    
    # Get the total number of sequences
    total_seqs = len(data_seqs)
    
    # Calculate how many sequences make up the %
    num_sample = max(1, int(total_seqs * sample_fraction))
    print(num_sample)
    
    # Randomly sample 10% of sequences and their corresponding IDs
    random.seed(seed)
    sampled_indices = random.sample(range(total_seqs), num_sample)
    
    # Create the reduced lists for sequences and IDs
    sampled_seqs = [data_seqs[i] for i in sampled_indices]
    sampled_ids = [data_ids[i] for i in sampled_indices]

    return sampled_ids, sampled_seqs



def save_batch_embeddings(batch_ids, batch_embeddings, output_file):
    """Save a batch of embeddings to a JSON file."""
    batch_dict = {seq_id: embedding for seq_id, embedding in zip(batch_ids, batch_embeddings)}
    
    # Append the batch to the output file
    with open(output_file, 'a') as f:
        json.dump(batch_dict, f)
        f.write('\n')  # Add a newline for separation between batches



def get_mlm_last_layer_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
            # Ensure to request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Access the last hidden states from the hidden_states tuple
            last_hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states
            
            # Mean pooling across the sequence length dimension
            mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy() # Average residue level emebddings to get sequence level embedding
            embeddings.append(mean_pooled_output)
    
    return np.vstack(embeddings)  # Stack into a numpy array



def main(args):
    # Check if a GPU is available and get the correct device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load sequences from FASTA file
    data_ids, data_seqs = load_fasta_file(args.file)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Create the pipeline for feature extraction with the selected device
    embeddings = get_mlm_last_layer_embeddings(model, tokenizer, data_seqs, device)

    np.savez(f'{args.output_file}', embeddings=embeddings, id = data_ids)


if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Compute protein sequence embeddings')
    parser.add_argument('--file', type=str, required=True, help='Path to the FASTA file of sequences.')
    parser.add_argument('--model', type=str, default="gsgueglia/sm_protgpt2_alpha", help='Model name or path to use.')
    parser.add_argument('--device', type=int, default=-1, help='Device index (-1 for CPU, >=0 for GPU).')
    parser.add_argument('--output_file', type=str, default='embeddings.json', help='File to save the embeddings dictionary.')

    args = parser.parse_args()
    main(args)




# HOW TO USE
# REMEMBER to choose the sample fraction of interest (line 8)

# python get_embeddings.py --file ./no_match/no_match_alpha/regex_no_match_generated_sequences_alpha_1M_SW.fasta --device 5 --output_file ./embeddings/no_matched_embedding_alpha.npz
