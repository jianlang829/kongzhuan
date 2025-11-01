import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys
sys.path.append(r'.')#让Python解释器能够导入位于
from GATES import GATES
from utils import *
from Train_GATES import train_GATES

# 定义数据根目录
data_root = r'C:\Users\Admin\Documents\kongzhuan\kongzhuan\DLPFC'
section_id = '151674'
print('-----1-------')
alpha = 0.01
resolution = 1
counts_file = os.path.join(r"C:\Users\Admin\Documents\kongzhuan\kongzhuan\DLPFC\151674\RNA_counts.tsv")
coor_file = os.path.join(r"C:\Users\Admin\Documents\kongzhuan\kongzhuan\DLPFC\151674\position.tsv")
counts = pd.read_csv(counts_file, sep='\t', index_col=0)
print('-------1.5-----')
coor_df = pd.read_csv(coor_file, sep='\t')
counts.columns = ['Spot_'+str(x) for x in counts.columns]
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x', 'y']]
print('-------2-----')
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)
print('-------3-----')
plt.rcParams["figure.figsize"] = (5, 4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')

plt.savefig(f'alpha{alpha}_{resolution}_data1.png', dpi=100, bbox_inches='tight')

print('-------4-----')
used_barcode = pd.read_csv(os.path.join(r"C:\Users\Admin\Documents\kongzhuan\kongzhuan\DLPFC\151674\used_barcodes.txt"),
                           sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode]

plt.rcParams["figure.figsize"] = (5, 4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
print('-------5-----')
plt.axis('off')
plt.savefig(f'alpha{alpha}_{resolution}_data2.png', dpi=100, bbox_inches='tight')

sc.pp.filter_genes(adata, min_cells=50)
print('-------6-----')
# Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# 读取人工注释的细胞类型标签
truth_file = os.path.join(data_root, section_id, '_truth.txt')
if os.path.exists(truth_file):
    Ann_df = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
else:
    print(f"Warning: Truth file for {section_id} not found. Skipping...")
print('-------7-----')
Cal_Spatial_Net(adata, rad_cutoff=50)
Stats_Spatial_Net(adata, save=f'alpha{alpha}_{resolution}_Stereo-seq_Mouse_NumberOfNeighbors.png', plt_show=False)

Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=True)
print('-------8-----')
adata = train_GATES(adata, mod='spatial-similarity', alpha=alpha, n_epochs=500, verbose=False, random_seed=2020)
sc.pp.neighbors(adata, use_rep='GATES')
print('-------9-----')
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=resolution)

# 计算 SC 和 DB 分数
# 使用 UMAP 嵌入表示作为输入
print('-------10-----')
umap_embedding = adata.obsm['X_umap']
louvain_labels = adata.obs['louvain'].astype(int)


# 计算 Silhouette Coefficient

sc_score = silhouette_score(umap_embedding, louvain_labels)
sc_score = round(sc_score, 2)
print('-------11-----')

# 计算 Davies-Bouldin Index

db_score = davies_bouldin_score(umap_embedding, louvain_labels)
db_score = round(db_score, 2)
print('-------12-----')
plt.rcParams["figure.figsize"] = (5, 4)
sc.pl.embedding(adata, basis="spatial", color="louvain", s=6, show=False, title='Ours')
plt.axis('off')
print('-------13-----')

import scanpy as sc
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 4)
sc.pl.embedding(adata, basis="spatial", color="louvain", s=6, show=False, title='Ours')
plt.axis('off')

x_min, x_max, y_min, y_max = 10100, 10721, 13093, 13810  # 注意 y 的顺序：下小上大
plt.xlim(x_min, x_max)

plt.tight_layout()

print('-------15-----')
sc.pl.umap(adata, color='louvain', title=f'Ours SC{sc_score} DB{db_score}', show=False)
# 保存图片
plt.savefig(f'alpha{alpha}_{resolution}_STAGATE_umap.png', dpi=100, bbox_inches='tight')

# 轨迹推断：使用 PAGA
print('-------16-----')
sc.tl.paga(adata)  # 计算PAGA
sc.pl.paga(adata, color='louvain', show=False, title=f'Ours SC:{sc_score} DB:{db_score}')  # 可视化PAGA图

plt.savefig(f'alpha{alpha}_{resolution}_STAGATE_paga.png', dpi=100, bbox_inches='tight')
print('-------17-----')
sc.pp.pca(adata, n_comps=3)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.louvain(adata, resolution=resolution)
sc.tl.umap(adata)

umap_embedding = adata.obsm['X_umap']
louvain_labels = adata.obs['louvain'].astype(int)

print('-------18-----')
# 计算 Silhouette Coefficient
sc_score = silhouette_score(umap_embedding, louvain_labels)
sc_score = round(sc_score, 2)

# 计算 Davies-Bouldin Index

print('-------19-----')
db_score = davies_bouldin_score(umap_embedding, louvain_labels)
db_score = round(db_score, 2)
print(f'Davies-Bouldin Index: {db_score}')


print('-------20-----')
plt.rcParams["figure.figsize"] = (5, 4)
sc.pl.embedding(adata, basis="spatial", color="louvain", s=6, show=False, title='SCANPY')
plt.axis('off')
#

plt.savefig(f'alpha{alpha}_{resolution}_SCANPY_cls_domain.png', dpi=100, bbox_inches='tight')

print('-------21-----')
sc.pl.umap(adata, color='louvain', title=f'SCANPY SC:{sc_score} DB:{db_score}', show=False)

# 保存图片
plt.savefig(f'alpha{alpha}_{resolution}_SCANPY_umap.png', dpi=100, bbox_inches='tight')
# 显示图片

print('-------22-----')

# 轨迹推断：使用 PAGA

sc.tl.paga(adata)  # 计算PAGA
sc.pl.paga(adata, color='louvain', show=False, title=f'SCANPY SC:{sc_score} DB:{db_score}')  # 可视化PAGA图
# 
print('-------23-----')