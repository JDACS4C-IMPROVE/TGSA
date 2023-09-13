from pathlib import Path

import numpy as np
import pandas as pd

from Functions import replace_ccl_name

fdir = Path(__file__).resolve().parent

# pd.set_option('display.max_columns', None)

# omics_data_dir = '../../Data_Curation_final/Curated_CCLE_Multiomics_files/'
omics_data_dir = fdir/'../../Data_Curation_final/Curated_CCLE_Multiomics_files' # ap

# benchmark_data_dir = '../CSA_Data/'
benchmark_data_dir = fdir/'../csa_data' # ap
x_data_dir = benchmark_data_dir/'x_data'  # ap



# Load cell line information
# ccl_info = pd.read_csv(benchmark_data_dir + 'ccl_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
#                  header=0, index_col=None, low_memory=False)
ccl_info = pd.read_csv(benchmark_data_dir/'ccl_info.txt', sep='\t', engine='c', na_values=['na', '-', ''],
                 header=0, index_col=None, low_memory=False)
# TODO: consider renaming improve_sample_id to imp_sample_id
id = np.where(ccl_info.id_source == 'Cellosaurus')[0]
ccl_info = ccl_info.iloc[id, :].loc[:, ['improve_sample_id', 'other_id']].drop_duplicates()

# Load gene expression data
# ge = pd.read_csv(omics_data_dir + 'CCLE_AID_expression_full.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
ge = pd.read_csv(omics_data_dir/'CCLE_AID_expression_full.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
ge = ge.drop_duplicates()
ge = replace_ccl_name(data=ge, keep_id=[0, 1, 2], ccl_info=ccl_info)
ge.iloc[0, 0] = ''
# ge.to_csv(benchmark_data_dir + 'cancer_gene_expression.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
ge.to_csv(x_data_dir/'cancer_gene_expression.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
ge = None

# Load copy number data
# cn = pd.read_csv(omics_data_dir + 'CCLE_AID_gene_cn.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
cn = pd.read_csv(omics_data_dir/'CCLE_AID_gene_cn.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
cn = cn.drop_duplicates()
cn = replace_ccl_name(data=cn, keep_id=[0, 1, 2], ccl_info=ccl_info)
cn.iloc[0, 0] = ''
# cn.to_csv(benchmark_data_dir + 'cancer_copy_number.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
cn.to_csv(x_data_dir/'cancer_copy_number.txt', header=False, index=False, sep='\t', line_terminator='\r\n')
cn = None

# Load discretized copy number data
# discretized_cn = pd.read_csv(omics_data_dir + 'CCLE_AID_gene_cn_discretized.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
discretized_cn = pd.read_csv(omics_data_dir/'CCLE_AID_gene_cn_discretized.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
discretized_cn = discretized_cn.drop_duplicates()
discretized_cn = replace_ccl_name(data=discretized_cn, keep_id=[0, 1, 2], ccl_info=ccl_info)
discretized_cn.iloc[0, 0] = ''
# discretized_cn.to_csv(benchmark_data_dir + 'cancer_discretized_copy_number.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
discretized_cn.to_csv(x_data_dir/'cancer_discretized_copy_number.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
discretized_cn = None

# Load miRNA expression data
# miRNA = pd.read_csv(omics_data_dir + 'CCLE_AID_miRNA_20180525.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
miRNA = pd.read_csv(omics_data_dir/'CCLE_AID_miRNA_20180525.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
miRNA = miRNA.drop_duplicates()
miRNA = replace_ccl_name(data=miRNA, keep_id=[0], ccl_info=ccl_info)
miRNA.iloc[0, 0] = ''
# miRNA.to_csv(benchmark_data_dir + 'cancer_miRNA_expression.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
miRNA.to_csv(x_data_dir/'cancer_miRNA_expression.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
miRNA = None

# Load protein expression data
# rppa = pd.read_csv(omics_data_dir + 'CCLE_AID_RPPA_20180123.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
rppa = pd.read_csv(omics_data_dir/'CCLE_AID_RPPA_20180123.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
rppa = rppa.drop_duplicates()
rppa = replace_ccl_name(data=rppa, keep_id=[0], ccl_info=ccl_info)
rppa.iloc[0, 0] = ''
# rppa.to_csv(benchmark_data_dir + 'cancer_RPPA.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
rppa.to_csv(x_data_dir/'cancer_RPPA.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
rppa = None

# Load DNA methylation data
# me = pd.read_csv(omics_data_dir + 'CCLE_AID_RRBS_TSS_1kb_20180614.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
me = pd.read_csv(omics_data_dir/'CCLE_AID_RRBS_TSS_1kb_20180614.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
me = me.drop_duplicates()
me = replace_ccl_name(data=me, keep_id=[0, 1, 2, 3], ccl_info=ccl_info)
me.iloc[0, 0] = ''
# me.to_csv(benchmark_data_dir + 'cancer_DNA_methylation.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
me.to_csv(x_data_dir/'cancer_DNA_methylation.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
me = None

# Load mutation count data
# mu_count = pd.read_csv(omics_data_dir + 'Mutation_AID_count.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
mu_count = pd.read_csv(omics_data_dir/'Mutation_AID_count.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=None, low_memory=False)
mu_count = mu_count.drop_duplicates()
mu_count = replace_ccl_name(data=mu_count, keep_id=[0, 1, 2], ccl_info=ccl_info)
mu_count.iloc[0, 0] = ''
# mu_count.to_csv(benchmark_data_dir + 'cancer_mutation_count.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
mu_count.to_csv(x_data_dir/'cancer_mutation_count.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
mu_count = None

# Load mutation data
# mu = pd.read_csv(omics_data_dir + 'Mutation_AID_binary2.csv', sep=',', engine='c',
#                         na_values=['na', '-', ''], header=None, index_col=0, low_memory=False)
mu = pd.read_csv(omics_data_dir/'Mutation_AID_binary2.csv', sep=',', engine='c',
                        na_values=['na', '-', ''], header=None, index_col=0, low_memory=False)
mu = mu.iloc[np.sort(np.setdiff1d(list(range(mu.shape[0])), [1])), :]
mu = mu.transpose()
mu = mu.drop_duplicates()
mu = replace_ccl_name(data=mu, keep_id=list(range(11)), ccl_info=ccl_info)
# mu.to_csv(benchmark_data_dir + 'cancer_mutation.txt', header=False, index=False, sep='\t',
#           line_terminator='\r\n')
mu.to_csv(x_data_dir/'cancer_mutation.txt', header=False, index=False, sep='\t',
          line_terminator='\r\n')
mu = None
