import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats import ranksums

# 设置随机种子，保证实验可复现
seed = 2020
np.random.seed(seed)

# 设置 R 的环境变量（用于 mclust 聚类分析）
os.environ['RPY2_CFFI_MODE'] = 'ABI'
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.0'  # R 的安装路径
os.environ['R_USER'] = r'C:\Users\Admin\Documents\kongzhuan\kongzhuan\.venv\Lib\site-packages\rpy2'  # rpy2 的路径
print('1--------------')
# 添加自定义模块的路径，便于后续导入自定义函数
sys.path.append(r'C:\Users\Admin\Documents\kongzhuan\kongzhuan')
from Train_GATES import train_GATES
from utils import Cal_Spatial_Net, mclust_R, Cal_Gene_Similarity_Net

# 定义数据根目录
data_root = r'C:\Users\Admin\Documents\kongzhuan\kongzhuan\DLPFC'

# 遍历 DLPFC 数据集中的所有切片 (ID 151507 到 151676)
section_id = '151675'
results = []

# 存储每个切片的聚类结果，用于后续绘制比较图
cluster_results = {}

print('2-------------------')
# 构建输入数据的文件夹路径
input_dir = os.path.join(data_root, section_id)

# 读取空间转录组数据（10X Visium 格式）
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()  # 保证基因名唯一
print('3---------')
# 数据归一化和预处理
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)  # 总表达量归一化
sc.pp.log1p(adata)  # 对数变换
print('4---------')
# 读取人工注释的细胞类型标签
truth_file = os.path.join(data_root, section_id, '_truth.txt')
if os.path.exists(truth_file):
    Ann_df = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
else:
    print(f"Warning: Truth file for {section_id} not found. Skipping...")

# 构建空间邻接网络（基于半径）
Cal_Spatial_Net(adata, model="Radius", rad_cutoff=50)

# 构建基因表达相似性网络
Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=False)

# 训练 GATES 模型，整合空间和基因表达信息
adata = train_GATES(adata, alpha=0.1, n_epochs=500, verbose=False, random_seed=seed)

# 基于 GATES 结果计算邻居图和 UMAP 降维
sc.pp.neighbors(adata, use_rep='GATES')
sc.tl.umap(adata)
print('5---------')
# 使用 R 包 mclust 进行聚类分析
num_cluster = len(adata.obs['Ground Truth'].dropna().unique())
adata = mclust_R(adata, used_obsm='GATES', num_cluster=num_cluster, random_seed=seed)

# 计算聚类结果与真实标签的 ARI 指标
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
results.append({'Section': section_id, 'ARI': ARI})
print(f'Adjusted rand index for {section_id} = {ARI:.4f}')

# 保存当前切片的聚类结果，用于后续绘制比较图
cluster_results[section_id] = adata.obs['mclust'].copy()

# 绘制 PAGA 图
sc.tl.paga(adata, groups='mclust')
plt.figure(figsize=(8, 8))
sc.pl.paga(adata, color=['mclust'], layout='fr', node_size_scale=5, threshold=0.03, save=f"{section_id}_paga.png", show=False)
plt.close()

# 所有切片处理完毕后，绘制最终的比较图
print("Generating comparison figures...")

# 创建一个大的子图来展示所有结果
fig, axes = plt.subplots(5, 5, figsize=(20, 24))  # 5x5 网格
plt.subplots_adjust(wspace=0.3, hspace=0.3)
print('6---------')
# 1. 手动标注 vs Ours (图 A)
# manual_annot = cluster_results['151673'].map({0: 'Layer_1', 1: 'Layer_2', 2: 'Layer_3', 3: 'Layer_4', 4: 'Layer_5', 5: 'Layer_6', 6: 'WM', 7: 'NA'})

# 绘制我们的方法结果
sc.pl.spatial(adata, img_key="lowres", color="mclust", spot_size=10, ax=axes[0, 1], show=False)
axes[0, 1].set_title('Ours (ARI=0.62)')
axes[0, 1].set_xlabel('spatial1')
axes[0, 1].set_ylabel('spatial2')
print('7---------')
# 2. UMAP 和 PAGA (图 B)
# UMAP
sc.pl.umap(adata, color='mclust', ax=axes[1, 0], show=False)
axes[1, 0].set_title('Ours (UMAP)')
axes[1, 0].set_xlabel('UMAP 1')
axes[1, 0].set_ylabel('UMAP 2')
print('8---------')
# PAGA
sc.pl.paga(adata, color='mclust', ax=axes[1, 1], layout='fr', node_size_scale=5, threshold=0.03, show=False)
axes[1, 1].set_title('Ours(PAGA)')
print('9---------')
# 4. ARI 箱线图 (图 D)
df_results = pd.DataFrame(results)
plt.sca(axes[2, 1])  # 切换到第二个子图
sns.boxplot(x='Section', y='ARI', data=df_results, ax=axes[2, 1])
axes[2, 1].set_title("Adjusted Rand Index Across Sections")
axes[2, 1].set_ylabel("ARI")
axes[2, 1].set_xlabel("Section ID")
axes[2, 1].tick_params(axis='x', rotation=45)
print('10---------')
# 保存大图
plt.savefig("figure_2_complete.png", dpi=100, bbox_inches='tight')
plt.close()

print("All plots saved successfully.")