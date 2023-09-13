import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

omics_data_dir = '../../Data_Curation_final/Curated_CCLE_Multiomics_files/'
auxiliary_data_dir = '../auxiliary_data/'

# Load gene expression data
ge = pd.read_csv(omics_data_dir + '/CCLE_AID_expression_full.csv', sep=',', engine='c', na_values=['na', '-', ''],
                 header=None, index_col=0, low_memory=False)
ge_ccl = ge.index[3:]
ge = None

# Load mutation count data
mu_count = pd.read_csv(omics_data_dir + '/Mutation_AID_count.csv', sep=',', engine='c', na_values=['na', '-', ''],
                       header=None, index_col=0, low_memory=False)
mu_ccl = mu_count.index[3:]
mu_count = None

# Load copy number data
cnv = pd.read_csv(omics_data_dir + '/CCLE_AID_gene_cn.csv', sep=',', engine='c', na_values=['na', '-', ''], header=None,
                  index_col=0, low_memory=False)
cnv_ccl = cnv.index[3:]
cnv = None

# Load DNA methylation data
me = pd.read_csv(omics_data_dir + '/CCLE_AID_RRBS_TSS_1kb_20180614.csv', sep=',', engine='c', na_values=['na', '-', ''],
                 header=None, index_col=0, low_memory=False)
me_ccl = me.index[4:]
me = None

ccl_ge_mu_cnv = np.intersect1d(np.intersect1d(ge_ccl, mu_ccl), cnv_ccl)
ccl_ge_mu_cnv_me = np.intersect1d(np.intersect1d(ge_ccl, mu_ccl), np.intersect1d(cnv_ccl, me_ccl))

pd.DataFrame(ccl_ge_mu_cnv).to_csv(auxiliary_data_dir + 'ccl_ge_mu_cnv.txt', header=False, index=False, sep='\t',
                                   line_terminator='\r\n')
pd.DataFrame(ccl_ge_mu_cnv_me).to_csv(auxiliary_data_dir + 'ccl_ge_mu_cnv_me.txt', header=False, index=False, sep='\t',
                                   line_terminator='\r\n')
