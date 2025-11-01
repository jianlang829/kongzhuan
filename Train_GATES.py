import numpy as np
import scipy.sparse as sp  # 导入稀疏矩阵处理库
from GATES import GATES  # 导入GATES模型
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为，使用TensorFlow 1.x的API
import pandas as pd
import scanpy as sc  # 导入单细胞分析库


def train_GATES(adata, hidden_dims=[512, 30], alpha=0, n_epochs=500, lr=0.0001, key_added='GATES',
                  gradient_clipping=5, nonlinear=True, weight_decay=0.0001, verbose=True,
                  random_seed=2024, pre_labels=None, pre_resolution=0.2, mod='spatial-similarity',
                  save_attention=False, save_loss=False, save_reconstrction=False):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    alpha
        The weight of cell type-aware spatial neighbor network.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    nonlinear
        If True, the nonlinear avtivation is performed.
    weight_decay
        Weight decay for AdamOptimizer.
    pre_labels
        The key in adata.obs for the manually designate the pre-clustering results. Only used when alpha>0.
    pre_resolution
        The resolution parameter of sc.tl.louvain for the pre-clustering. Only used when alpha>0 and per_labels==None.
    save_attention
        If True, the weights of the attention layers are saved in adata.uns['STAGATE_attention']
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].

    Returns
    -------
    AnnData
    """

    # 重置TensorFlow计算图，确保干净的环境
    tf.reset_default_graph()
    # 设置随机种子，确保结果可复现
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # 检查是否有高变基因信息，如果有则只使用高变基因进行训练
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    # 将AnnData对象转换为DataFrame，便于后续处理
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)

    # 输出输入数据的大小（如果verbose设置为True）
    if verbose:
        print('Size of Input: ', adata_Vars.shape)

    # 获取细胞列表和创建细胞ID到索引的映射字典
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    # 检查必要的网络数据是否存在
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    if 'Gene_Similarity_Net' not in adata.uns.keys():
        raise ValueError("Gene_Similarity_Net is not existed! Run Cal_Gene_Similarity_Net first!")

    # 获取空间网络数据
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()

    # 将细胞名称映射为索引
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    # 创建空间邻接矩阵（稀疏格式）
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G_tf = prepare_graph_data(G)  # 准备图数据用于TensorFlow

    # 初始化GATES模型
    trainer = GATES(hidden_dims=[X.shape[1]] + hidden_dims, alpha=alpha,
                    n_epochs=n_epochs, lr=lr, gradient_clipping=gradient_clipping,
                    nonlinear=nonlinear, weight_decay=weight_decay, verbose=verbose,
                    random_seed=random_seed)

    # 根据不同的模式进行训练
    if mod == 'spatial-prune':
        # 空间修剪模式：基于空间邻居网络的修剪版本
        if alpha == 0:
            # 如果alpha为0，只使用原始空间网络
            trainer(G_tf, G_tf, X)
            embeddings, attentions, loss, ReX = trainer.infer(G_tf, G_tf, X)
        else:
            # 使用修剪后的空间网络
            G_df = Spatial_Net.copy()
            if pre_labels == None:
                # 如果没有提供预聚类标签，使用louvain算法进行预聚类
                if verbose:
                    print('------Pre-clustering using louvain with resolution=%.2f' % pre_resolution)
                sc.tl.pca(adata, svd_solver='arpack')  # 先进行PCA降维
                sc.pp.neighbors(adata)  # 构建细胞邻居关系
                sc.tl.louvain(adata, resolution=pre_resolution, key_added='expression_louvain_label')  # 使用louvain算法聚类
                pre_labels = 'expression_louvain_label'

            # 根据预聚类标签修剪空间网络，只保留同一聚类内的连接
            prune_G_df = prune_spatial_Net(G_df, adata.obs[pre_labels])
            prune_G_df['Cell1'] = prune_G_df['Cell1'].map(cells_id_tran)
            prune_G_df['Cell2'] = prune_G_df['Cell2'].map(cells_id_tran)

            # 创建修剪后的空间邻接矩阵
            prune_G = sp.coo_matrix((np.ones(prune_G_df.shape[0]), (prune_G_df['Cell1'], prune_G_df['Cell2'])))
            prune_G_tf = prepare_graph_data(prune_G)
            prune_G_tf = (prune_G_tf[0], prune_G_tf[1], G_tf[2])  # 确保模型在训练和推理过程中处理的图数据维度一致

            # 使用原始空间网络和修剪后的网络进行训练
            trainer(G_tf, prune_G_tf, X)
            embeddings, attentions, loss, ReX = trainer.infer(G_tf, prune_G_tf, X)

    elif mod == 'spatial-similarity':
        # 空间相似性模式：结合空间网络和基因相似性网络
        if alpha == 0:
            # 如果alpha为0，只使用原始空间网络
            trainer(G_tf, G_tf, X)
            embeddings, attentions, loss, ReX = trainer.infer(G_tf, G_tf, X)
        else:
            # 使用基因相似性网络作为修剪网络
            Similarity_Net = adata.uns['Gene_Similarity_Net']
            G_sn = Similarity_Net.copy()
            G_sn['Cell1'] = G_sn['Cell1'].map(cells_id_tran)
            G_sn['Cell2'] = G_sn['Cell2'].map(cells_id_tran)

            # 检查映射后是否有NaN值
            if G_sn['Cell1'].isnull().any() or G_sn['Cell2'].isnull().any():
                print("Error: Mapping contains NaN values")
                print(G_sn[G_sn['Cell1'].isnull() | G_sn['Cell2'].isnull()])  # 打印出含有NaN的行

            # 创建基因相似性邻接矩阵
            G_ = sp.coo_matrix((np.ones(G_sn.shape[0]), (G_sn['Cell1'], G_sn['Cell2'])),
                               shape=(adata.n_obs, adata.n_obs))
            G_sn = prepare_graph_data(G_)

            # 使用空间网络和基因相似性网络进行训练
            trainer(G_tf, G_sn, X)
            embeddings, attentions, loss, ReX = trainer.infer(G_tf, G_sn, X)

    elif mod == 'prune-similarity':
        # 修剪相似性模式：结合修剪后的空间网络和基因相似性网络
        G_df = Spatial_Net.copy()
        if pre_labels == None:
            # 如果没有提供预聚类标签，使用louvain算法进行预聚类
            if verbose:
                print('------Pre-clustering using louvain with resolution=%.2f' % pre_resolution)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata)
            sc.tl.louvain(adata, resolution=pre_resolution, key_added='expression_louvain_label')
            pre_labels = 'expression_louvain_label'

        # 根据预聚类标签修剪空间网络
        prune_G_df = prune_spatial_Net(G_df, adata.obs[pre_labels])
        prune_G_df['Cell1'] = prune_G_df['Cell1'].map(cells_id_tran)
        prune_G_df['Cell2'] = prune_G_df['Cell2'].map(cells_id_tran)

        # 创建修剪后的空间邻接矩阵
        prune_G = sp.coo_matrix((np.ones(prune_G_df.shape[0]), (prune_G_df['Cell1'], prune_G_df['Cell2'])))
        prune_G_tf = prepare_graph_data(prune_G)
        prune_G_tf = (prune_G_tf[0], prune_G_tf[1], G_tf[2])  # 确保模型在训练和推理过程中处理的图数据维度一致

        if alpha == 0:
            # 如果alpha为0，只使用修剪后的空间网络
            trainer(prune_G_tf, prune_G_tf, X)
            embeddings, attentions, loss, ReX = trainer.infer(prune_G_tf, prune_G_tf, X)
        else:
            # 使用修剪后的空间网络和基因相似性网络
            Similarity_Net = adata.uns['Gene_Similarity_Net']
            G_sn = Similarity_Net.copy()
            G_sn['Cell1'] = G_sn['Cell1'].map(cells_id_tran)
            G_sn['Cell2'] = G_sn['Cell2'].map(cells_id_tran)

            # 检查映射后是否有NaN值
            if G_sn['Cell1'].isnull().any() or G_sn['Cell2'].isnull().any():
                print("Error: Mapping contains NaN values")
                print(G_sn[G_sn['Cell1'].isnull() | G_sn['Cell2'].isnull()])  # 打印出含有NaN的行

            # 创建基因相似性邻接矩阵
            G_ = sp.coo_matrix((np.ones(G_sn.shape[0]), (G_sn['Cell1'], G_sn['Cell2'])),
                               shape=(adata.n_obs, adata.n_obs))
            G_sn = prepare_graph_data(G_)

            # 使用修剪后的空间网络和基因相似性网络进行训练
            trainer(prune_G_tf, G_sn, X)
            embeddings, attentions, loss, ReX = trainer.infer(prune_G_tf, G_sn, X)
    else:
        # 如果提供了无效的模式，抛出错误
        raise ValueError("mod: spatial-prune, spatial-similarity, prune-similarity")

    # 将嵌入结果保存到AnnData对象中
    cell_reps = pd.DataFrame(embeddings)
    cell_reps.index = cells
    adata.obsm[key_added] = cell_reps.loc[adata.obs_names, ].values

    # 根据设置保存额外信息
    if save_attention:
        adata.uns['STAGATE_attention'] = attentions  # 保存注意力权重
    if save_loss:
        adata.uns['STAGATE_loss'] = loss  # 保存训练损失
    if save_reconstrction:
        ReX = pd.DataFrame(ReX, index=X.index, columns=X.columns)
        ReX[ReX < 0] = 0  # 将负值设为0
        adata.layers['STAGATE_ReX'] = ReX.values  # 保存重构的表达谱

    return adata


def prune_spatial_Net(Graph_df, label):
    """
    根据预聚类标签修剪空间网络，只保留同一聚类内的连接

    参数:
    Graph_df: DataFrame, 包含空间网络信息的数据框，每行表示一条边
    label: Series, 包含每个细胞对应的聚类标签

    返回:
    DataFrame, 修剪后的空间网络
    """

    # 创建细胞到标签的映射字典
    pro_labels_dict = dict(zip(list(label.index), label))

    # 为每条边添加源细胞和目标细胞的标签
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)

    # 只保留源细胞和目标细胞标签相同的边
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'],]

    return Graph_df


def prepare_graph_data(adj):
    """
    准备图数据用于TensorFlow模型

    参数:
    adj: 稀疏矩阵, 邻接矩阵

    返回:
    tuple, 包含(索引，数据，形状)的元组，用于创建TensorFlow的稀疏张量
    """
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # 添加自环，确保每个节点都连接到自身

    # 确保邻接矩阵是COO格式
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()

    adj = adj.astype(np.float32)  # 转换为float32类型

    # 创建稀疏张量所需的索引和数据
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)


def recovery_Imputed_Count(adata, size_factor):
    """
    从重构的表达谱中恢复细胞计数

    参数:
    adata: AnnData对象
    size_factor: Series或DataFrame, 包含每个细胞的大小因子

    返回:
    AnnData对象，包含恢复的计数数据
    """
    assert ('ReX' in adata.uns)  # 确保重构的表达谱存在
    temp_df = adata.uns['ReX'].copy()
    sf = size_factor.loc[temp_df.index]  # 获取对应细胞的大小因子

    # 从对数空间转换回线性空间
    temp_df = np.expm1(temp_df)

    # 应用大小因子调整计数
    temp_df = (temp_df.T * sf).T

    # 保存恢复的计数到AnnData对象
    adata.uns['ReX_Count'] = temp_df
    return adata
