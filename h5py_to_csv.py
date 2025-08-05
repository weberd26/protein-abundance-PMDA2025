import h5py
import pandas as pd

# Path to your HDF5 file
h5_file_path = "protein.sequence.embeddings.v12.0.h5"
output_csv_path = "protein_sequence_embeddings.csv"

# Store all data here
all_protein_embeddings = []

with h5py.File(h5_file_path, "r") as f:
    for species_id in f["/species"]:
        group = f["/species"][species_id]
        if "proteins" in group and "embeddings" in group:
            proteins = [p.decode("utf-8") for p in group["proteins"][:]]
            embeddings = group["embeddings"][:]
            for protein, emb in zip(proteins, embeddings):
                all_protein_embeddings.append([protein] + emb.tolist())

# Create DataFrame
df = pd.DataFrame(all_protein_embeddings)
df.columns = ["protein_id"] + [f"dim_{i}" for i in range(df.shape[1] - 1)]

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"✅ CSV file saved to: {output_csv_path}")

