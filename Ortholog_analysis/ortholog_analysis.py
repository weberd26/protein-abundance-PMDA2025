import pandas as pd
import numpy as np 


# %% Inputs

thresh_integrated_score = 0
min_samples_organism = 500

# %% Load data

file_path = './protein-abundance-PMDA2025/all_organisms_filtered_without_M.musculus_KIDNEY.parquet'
df = pd.read_parquet(file_path)

df['integrated_score'] = np.where(df['is_integrated'] == True, df['quality_score'], np.nan)
df['non_integrated_score'] = np.where(df['is_integrated'] == False, df['quality_score'], np.nan)

species_breakdown_updated = df.groupby('organism_name').agg(
    total_protein_entries=('EnsemblProteinID', 'count'),
    unique_proteins_name=('UniprotEntryName', 'nunique'),
    unique_proteins_id=('UniprotAccession', 'nunique'),
    avg_quality_score=('quality_score', 'mean'),
    avg_integrated_quality=('integrated_score', 'mean'),
    avg_non_integrated_quality=('non_integrated_score', 'mean')
)

species_breakdown_updated = species_breakdown_updated.sort_values(by='total_protein_entries', ascending=False)
score_cols = ['avg_quality_score', 'avg_integrated_quality', 'avg_non_integrated_quality']
species_breakdown_updated[score_cols] = species_breakdown_updated[score_cols].round(2)

print(species_breakdown_updated.head(5))

# %% Preprocessing

filter_used = (df['uniprot_status'] == 'available') & (df['Sequence'].str.len() > 10) & ~np.isnan(df['integrated_score'])  
filter_used_thres = filter_used & (df['integrated_score'] > thresh_integrated_score)
df_clean = df[filter_used_thres]

# Check histogram integrated score
import matplotlib.pyplot as plt
plt.hist(df_clean['integrated_score'], bins=20, edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()      

# Get tissues and organsims (throw out if whole organ is unavailable)
species = df_clean['organism_name'].unique()
spec_filtered = []
for spec in species:
    df_spec = df_clean[df_clean['organism_name'] == spec]
    if len(df_spec) > min_samples_organism:
        if len(df_spec[df_spec['sample_organ'] == 'WHOLE_ORGANISM']) > 2:
            spec_filtered.append(spec)
    
print(f'ORGANISMS DROPPED: {len(species) - len(spec_filtered)}')

# %% Predictive power matrix (similarity scores) for organisms (~15 min)

from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from scipy.stats import spearmanr

def aggregate_orthologs_mean(ids, values):
      u, inv = np.unique(ids, return_inverse=True) # u unique ids 
      sums   = np.bincount(inv, weights=values)
      counts = np.bincount(inv)
      means  = sums / counts
      return u, means
    
min_matches_matrix = 10

r2_matrix = np.zeros((len(spec_filtered), len(spec_filtered)))
r2_matrix_st = np.zeros((len(spec_filtered), len(spec_filtered)))
mse_matrix = np.full((len(spec_filtered), len(spec_filtered)), np.inf)
mse_matrix_st = np.full((len(spec_filtered), len(spec_filtered)), np.inf)
corr_matrix = np.zeros((len(spec_filtered), len(spec_filtered)))
spear_corr_matrix = np.zeros((len(spec_filtered), len(spec_filtered)))

for s1, spec1 in tqdm(enumerate(spec_filtered)):
    for s2, spec2 in enumerate(spec_filtered):
        df_spec1 = df_clean[df_clean['organism_name'] == spec1]
        df_spec2 = df_clean[df_clean['organism_name'] == spec2]
        
        df_spec1_whole = df_spec1[df_spec1['sample_organ'] == 'WHOLE_ORGANISM']
        df_spec2_whole = df_spec2[df_spec2['sample_organ'] == 'WHOLE_ORGANISM']
        
        abun1_raw = np.log(df_spec1_whole['abundance'].to_numpy())
        mean1 = np.mean(abun1_raw)
        std1 = np.std(abun1_raw)
        abun2_raw = np.log(df_spec2_whole['abundance'].to_numpy())
        mean2 = np.mean(abun2_raw)
        std2 = np.std(abun2_raw)
               
        id1_raw = df_spec1_whole['nog_id'].to_numpy()
        id2_raw = df_spec2_whole['nog_id'].to_numpy()
               
        mask1 = (id1_raw != None)
        mask2 = (id2_raw != None)
        
        abun1 = abun1_raw[mask1]
        abun2 = abun2_raw[mask2]
        abun1_st = (abun1 - mean1) / std1
        abun2_st = (abun2 - mean2) / std2
        
        id1 = id1_raw[mask1]
        id2 = id2_raw[mask2]
        
        id1_agg, abun1_agg = aggregate_orthologs_mean(id1, abun1)
        id2_agg, abun2_agg = aggregate_orthologs_mean(id2, abun2)
        _, abun1_agg_st = aggregate_orthologs_mean(id1, abun1_st)
        _, abun2_agg_st = aggregate_orthologs_mean(id2, abun2_st)
         
        i_idx, j_idx = np.nonzero(id1_agg[:, None] == id2_agg)
        matches_np = np.column_stack((i_idx, j_idx))
        
        if len(matches_np) > 2:
           
            r2 = r2_score(abun1_agg[i_idx], abun2_agg[j_idx])
            r2_st = r2_score(abun1_agg_st[i_idx], abun2_agg_st[j_idx])
            
            mse = mean_squared_error(abun1_agg[i_idx], abun2_agg[j_idx])
            mse_st = mean_squared_error(abun1_agg_st[i_idx], abun2_agg_st[j_idx])
            corr = np.corrcoef(abun1_agg[i_idx], abun2_agg[j_idx])
            spear = spearmanr(abun1_agg[i_idx], abun2_agg[j_idx])[0]
            
            corr_matrix[s1,s2] = corr[0,1]
            spear_corr_matrix[s1,s2] = spear
            r2_matrix[s1,s2] = r2
            r2_matrix_st[s1,s2] = r2_st
            mse_matrix[s1,s2] = mse
            mse_matrix[s1,s2] = mse_st
            
# mean and standard ab whole species
mean_ab_spec = []
std_ab_spec = []
for _, spec in enumerate(spec_filtered):
    df_spec = df_clean[df_clean['organism_name'] == spec]
   
    df_spec_whole = df_spec[df_spec['sample_organ'] == 'WHOLE_ORGANISM']
   
    abun_raw = df_spec_whole['abundance'].to_numpy()
    mean_ab_spec.append(np.mean(abun_raw))
    std_ab_spec.append(np.std(abun_raw))

# %% Save stuff

np.save('./species_filtered.npy', spec_filtered)
np.save('./corr_matrix.npy', corr_matrix)
np.save('./spear_matrix.npy', spear_corr_matrix)
np.save('./r2_matrix.npy', r2_matrix)
np.save('./mse_matrix.npy', mse_matrix)

# %% MSE




    

# %%      Testing

df_test = df_spec[df_spec['sample_organ'] == 'KIDNEY']

# corr_matrix[1,:]

# np.array(spec_filtered)[np.argsort(corr_matrix[1,:])]


# # example plot
# s1 = 1
# s2 = 3
# spec1 = spec_filtered[s1]
# spec2 = spec_filtered[s2]

# df_spec1 = df_clean[df_clean['organism_name'] == spec1]
# df_spec2 = df_clean[df_clean['organism_name'] == spec2]

# df_spec1_whole = df_spec1[df_spec1['sample_organ'] == 'WHOLE_ORGANISM']
# df_spec2_whole = df_spec2[df_spec2['sample_organ'] == 'WHOLE_ORGANISM']

# abun1_raw = np.log(df_spec1_whole['abundance'].to_numpy())
# mean1 = np.mean(abun1_raw)
# std1 = np.std(abun1_raw)
# abun2_raw = np.log(df_spec2_whole['abundance'].to_numpy())
# mean2 = np.mean(abun2_raw)
# std2 = np.std(abun2_raw)
       
# id1_raw = df_spec1_whole['nog_id'].to_numpy()
# id2_raw = df_spec2_whole['nog_id'].to_numpy()
       
# mask1 = (id1_raw != None)
# mask2 = (id2_raw != None)

# abun1 = abun1_raw[mask1]
# abun2 = abun2_raw[mask2]
# abun1_st = (abun1 - mean1) / std1
# abun2_st = (abun2 - mean2) / std2

# id1 = id1_raw[mask1]
# id2 = id2_raw[mask2]

# id1_agg, abun1_agg = aggregate_orthologs_mean(id1, abun1)
# id2_agg, abun2_agg = aggregate_orthologs_mean(id2, abun2)
# _, abun1_agg_st = aggregate_orthologs_mean(id1, abun1_st)
# _, abun2_agg_st = aggregate_orthologs_mean(id2, abun2_st)
 
# i_idx, j_idx = np.nonzero(id1_agg[:, None] == id2_agg)
# matches_np = np.column_stack((i_idx, j_idx))

# r2 = r2_score(abun1_agg[i_idx], abun2_agg[j_idx])
# r2_st = r2_score(abun1_agg_st[i_idx], abun2_agg_st[j_idx])

# mse = mean_squared_error(abun1_agg[i_idx], abun2_agg[j_idx])
# mse_st = mean_squared_error(abun1_agg_st[i_idx], abun2_agg_st[j_idx])
# corr = np.corrcoef(abun1_agg[i_idx], abun2_agg[j_idx])


# y = abun1_agg[i_idx]
# y_pred = abun2_agg[j_idx]

# fig, ax = plt.subplots()

# # scatter
# ax.scatter(y, y_pred, alpha=0.6)

# lims = [np.min([y.min(), y_pred.min()]), np.max([y.max(), y_pred.max()])]
# ax.plot(lims, lims, linestyle='--')     # dashed reference line
# ax.set_xlim(lims)
# ax.set_ylim(lims)

# ax.set_xlabel('abundance O1 [ppm]')
# ax.set_ylabel('abundance O1 [ppm]')
# ax.set_title('O1 vs O2')
# ax.set_aspect('equal', adjustable='box')  # keeps the diagonal at 45°

# plt.show()














