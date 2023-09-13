from pathlib import Path  # pathlib introduced in Python 3.4 (standard, intuitive, powerful as compared to os.path)

import numpy as np
import pandas as pd

fdir = Path(__file__).resolve().parent  # absolute path to parent dir (can call this func from anywhere)

# pd.set_option('display.max_columns', None)

# drug_data_dir = '../../Drug_Data/'
drug_data_dir = fdir/'../../Drug_Data' # ap

# benchmark_data_dir = '../CSA_Data/'
benchmark_data_dir = fdir/'../csg_data' # ap
x_data_dir = benchmark_data_dir/'x_data'  # ap



# Load drug fingerprint data
# df = pd.read_parquet(drug_data_dir + 'ecfp4_nbits512.parquet', engine='pyarrow')
df = pd.read_parquet(drug_data_dir/'ecfp4_nbits512.parquet', engine='pyarrow') # ap
print("\nECFP4", df.shape)
df = df.drop_duplicates()
df.index = df.improve_chem_id
df = df.iloc[:, 1:]
print("ECFP4", df.shape)

# Load drug descriptor data
# dd = pd.read_parquet(drug_data_dir + 'mordred.parquet', engine='pyarrow')
dd = pd.read_parquet(drug_data_dir/'mordred.parquet', engine='pyarrow') # ap
print("\nMordred", dd.shape)
dd = dd.drop_duplicates()
dd.index = dd.improve_chem_id
dd = dd.iloc[:, 1:]
print("Mordred", dd.shape)

# Load drug info (metadata)
# import pdb; ipdb.set_trace()
drug_info = pd.read_csv(benchmark_data_dir/'drug_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
                        header=0, index_col=None, low_memory=False)
print("\nDrug info", drug_info.shape)
# Note that not the original canSMILES were used to calc ecfp and mordred!
ds = drug_info.loc[:, ['canSMILES', 'improve_chem_id']].drop_duplicates()
ds.index = ds.improve_chem_id  # TODO: consider renaming 'improve_chem_id' to 'imp_drug_id'
ds = ds.iloc[:, [0]]
id = np.argsort(ds.iloc[:, 0])  # Note! Same as ds.sort_values('canSMILES', ascending=True)
ds = ds.iloc[id, :]
print("Drug info", drug_info.shape)

# Note! This is same as below.
# common = list(set(df.index).intersection(set(ds.index))) # ap
# df_tmp = df[df.index.isin(common)] # ap
id = np.where(np.isin(df.index, ds.index))[0]  
df = df.iloc[id, :]
# df.equals(df_tmp) # ap

id = np.where(np.isin(dd.index, ds.index))[0]
dd = dd.iloc[id, :]

# ds.to_csv(benchmark_data_dir + 'drug_SMILES.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
# df.to_csv(benchmark_data_dir + 'drug_fingerprint.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
# dd.to_csv(benchmark_data_dir + 'drug_descriptor.txt', header=True, index=True, sep='\t', line_terminator='\r\n')

# ap
ds.to_csv(x_data_dir/'drug_SMILES.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
df.to_csv(x_data_dir/'drug_ecfp4_nbits512.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
dd.to_csv(x_data_dir/'drug_mordred_descriptor.txt', header=True, index=True, sep='\t', line_terminator='\r\n')

## ap
# ds.to_csv(benchmark_data_dir/'drug_smiles.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
# df.to_csv(benchmark_data_dir/'ecfp4_nbits512.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
# dd.to_csv(benchmark_data_dir/'mordred.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
