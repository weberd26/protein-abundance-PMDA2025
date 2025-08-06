import h5py # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]

# Load the CSV and extract the relevant STRING IDs
df_ids = pd.read_csv("data/DEF_PROCESSED_all_organisms_filtered_without_M.musculus_KIDNEY.csv")
relevant_ids = set(df_ids['string_external_id'])

print(f"Loaded {len(relevant_ids)} relevant STRING IDs")


filename = 'data/protein.network.embeddings.v12.0.h5'
rows = []
# relevant_ids_test = list(relevant_ids)[:5]  # Use the relevant IDs loaded from the CSV

with h5py.File(filename, 'r') as f:
    species_group = f['species']
    for species_id in species_group:
        group = species_group[species_id]
        if 'proteins' in group and 'embeddings' in group:
            protein_ids = group['proteins'][:]
            embeddings = group['embeddings'][:]
            # Decode protein IDs
            protein_ids = [pid.decode('utf-8') for pid in protein_ids]
            # Build a mapping from protein_id to embedding index for fast lookup
            pid_to_idx = {pid: idx for idx, pid in enumerate(protein_ids)}
            # Find intersection with relevant_ids
            matching_ids = relevant_ids.intersection(pid_to_idx.keys())
            for pid in matching_ids:
                idx = pid_to_idx[pid]
                emb = embeddings[idx]
                rows.append([species_id, pid] + emb.tolist())

# Build DataFrame
embedding_dim = len(rows[0]) - 2 if rows else 0
columns = ['species_id', 'protein_id'] + [f'emb_{i}' for i in range(embedding_dim)]
df = pd.DataFrame(rows, columns=columns)

print(df.head())

output_csv_path_network = "data/protein_network_embeddings.csv"
df.to_csv(output_csv_path_network, index=False)
print(f"✅ CSV file saved to: {output_csv_path_network}")