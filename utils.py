# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import sklearn.neighbors  # 用于构建近邻图
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances  # 用于计算样本间的相似度
from scipy.stats import pearsonr  # 用于计算皮尔逊相关系数
from sklearn.metrics.cluster import adjusted_rand_score


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    # 函数功能: 构建空间邻居网络, 基于细胞/点的空间坐标信息
    # 该函数通过计算细胞/点之间的空间距离构建邻居关系网络, 支持基于固定半径(Radius)或K近邻(KNN)两种方式

    # 确保模型参数为'Radius'或'KNN'中的一种
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    # 从adata中提取空间坐标信息
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index  # 设置坐标的索引为细胞/点的索引
    coor.columns = ['imagerow', 'imagecol']  # 命名坐标列

    # 如果使用Radius模型（基于距离阈值的连接）
    if model == 'Radius':
        # 构建基于半径的近邻模型, 所有在指定半径rad_cutoff内的点都被认为是邻居
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        # 计算每个点在指定半径内的所有邻居及其距离, indices存储邻居索引, distances存储对应距离
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        # 为每个点创建与其邻居的连接信息, 构建边的列表
        for it in range(indices.shape[0]):
            # 为每个细胞创建一个DataFrame, 包含自身索引(it)、邻居索引及到邻居的距离
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    # 如果使用KNN模型（基于K个最近邻的连接）
    if model == 'KNN':
        # 构建K近邻模型（+1是因为包括自身，自身总是最近的点）
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        # 计算每个点的K个最近邻及其距离, indices是二维数组, 每行对应一个点的K个最近邻索引
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        # 为每个点创建与其K个最近邻的连接信息
        for it in range(indices.shape[0]):
            # 为每个细胞创建一个DataFrame, 包含自身索引(it)重复K次、K个最近邻索引和对应距离
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    # 合并所有连接信息到一个统一的DataFrame
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 命名列: 源细胞、目标细胞、距离

    # 复制并处理连接信息, 创建最终的空间网络
    Spatial_Net = KNN_df.copy()
    # 移除自连接（距离为0的连接, 即细胞与自身的连接）
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    # 创建索引映射字典, 将数字索引(0,1,2...)转为原始细胞/点的标识符(实际的细胞ID)
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    # 将Cell1和Cell2的数字索引映射回原始标识符, 使网络更有可读性和实际意义
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    
    # 如果verbose为True，输出网络统计信息
    if verbose:
        # 显示网络中的边数和细胞数, 边数对应空间网络中的连接数
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        # 显示平均每个细胞的邻居数量, 用于评估网络连接密度
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    # 将构建的空间网络保存到adata对象的uns字典中, 键名为'Spatial_Net'
    # 这样其他函数可以通过adata.uns['Spatial_Net']访问该网络
    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata, save=False, plt_show=False):
    """
    统计并可视化空间网络的属性。
    
    参数:
    -----
    adata : AnnData对象
        包含空间网络信息的AnnData对象，需要先运行Cal_Spatial_Net
    save : bool 或 str
        是否保存图像，若为字符串则为保存路径
    plt_show : bool
        是否显示图像
        
    返回:
    -----
    无返回值，但会生成并可选择性保存/显示邻居数量分布图
    """
    import matplotlib.pyplot as plt  # 导入可视化库

    # 计算网络的边数和平均边数
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]  # 总边数（网络中的连接总数）
    Mean_edge = Num_edge / adata.shape[0]  # 平均每个细胞的连接数（总边数除以细胞数）

    # 统计每个节点的邻居数量分布
    # 第一步: 计算每个细胞有多少个邻居（通过统计每个细胞在Cell1列中出现的次数）
    # 第二步: 计算具有特定邻居数量的细胞比例（例如: 有多少细胞有3个邻居, 有多少有4个邻居等）
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]  # 转换为百分比（占总细胞数的比例）
    
    if plt_show:
        # 创建图形, 设置大小为3x2英寸
        fig, ax = plt.subplots(figsize=[3, 2])
        plt.ylabel('Percentage')  # Y轴标签：百分比, 表示拥有特定邻居数的细胞比例
        plt.xlabel('')  # X轴标签：留空, X轴默认显示邻居数量
        plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)  # 标题：邻居数量及其平均值

        # 绘制柱状图, X轴是邻居数量(plot_df.index), Y轴是对应细胞的比例(plot_df)
        ax.bar(plot_df.index, plot_df)

        # 如果需要保存图像
        if save:
            plt.tight_layout()  # 确保布局不重叠, 调整子图参数使图表元素不重叠
            plt.savefig(save, dpi=150, bbox_inches='tight')  # 保存为指定文件, dpi=150确保图像质量高
    


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    
    Parameters:
    -----
    adata : AnnData object
        Annotated data matrix.
    num_cluster : int
        Number of clusters to find.
    modelNames : str
        Model name for mclust (default 'EEE').
    used_obsm : str
        Key in adata.obsm to use for clustering.
    random_seed : int
        Random seed for reproducibility.

    Returns:
    -----
    adata : AnnData object
        Updated with 'mclust' column in adata.obs.
    """
    import numpy as np
    np.random.seed(random_seed)

    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    from rpy2.robjects.packages import importr

    # 加载 mclust 包
    mclust = importr('mclust')
    rmclust = mclust.Mclust

    # 设置 R 随机种子
    r_set_seed = robjects.r['set.seed']
    r_set_seed(random_seed)

    # 获取数据
    data = adata.obsm[used_obsm]

    with localconverter(default_converter):
        # 转换为 R 矩阵
        r_data = robjects.FloatVector(data.flatten())
        r_matrix = robjects.r['matrix'](r_data, nrow=data.shape[0], byrow=True)

        # 调用 Mclust 函数
        res = rmclust(r_matrix, G=num_cluster, modelNames=modelNames)
    
        # 提取分类结果
    try:
        idx = list(res.names).index('classification')
        classification = res[idx]
    except ValueError:
        raise KeyError("Field 'classification' not found in Mclust results")

    mclust_res = np.array(classification)

    # 将聚类结果保存到 adata.obs 中
    adata.obs['mclust'] = pd.Categorical(mclust_res.astype(int))  # 正确方式

    return adata


def Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=True):
    """
    计算细胞间基因表达的相似度，并构建稀疏矩阵。

    Parameters:
    -----------
    adata : AnnData
        包含基因表达数据的 AnnData 对象
    k_neighbors : int
        每个细胞选择最相似的邻居数量
    metric : str
        相似度度量方式，可选 'cosine', 'euclidean', 'correlation', 'pearson'

    Returns:
    --------
    KNN_df : pd.DataFrame
        稀疏矩阵，包含列 ['Cell1', 'Cell2', 'Distance']
    """

    # 首先检查数据中是否有已标记的高变异基因，如果有则只使用高变异基因进行计算，提高效率和准确性
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]  # 仅使用高变异基因
    else:
        adata_Vars = adata  # 使用全部基因
    # adata_Vars = adata
    
    # 将稀疏矩阵转为密集矩阵并创建DataFrame，以便计算相似度
    X = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)

    # 根据选择的度量方法计算细胞间的相似度矩阵
    if metric == 'cosine':
        # 余弦相似度：测量两个向量方向的相似性（不考虑大小）
        similarity_matrix = cosine_similarity(X)
    elif metric == 'euclidean':
        # 欧氏距离：计算两点间的直线距离，取负值将距离转为相似度（距离越小，相似度越高）
        similarity_matrix = -euclidean_distances(X)  # 将距离转为相似度，距离越小相似度越大
    elif metric == 'pearson':
        # 皮尔逊相关系数：测量两个变量之间的线性相关程度
        similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                # 计算每对细胞之间的皮尔逊相关系数（仅取相关系数值，不取p值）
                similarity_matrix[i, j] = pearsonr(X.iloc[i, :], X.iloc[j, :])[0]
    else:
        # 如果提供了未知的相似度度量方法，则抛出错误
        raise ValueError(f"未知的相似度度量: {metric}")

    # 创建列表存储每个细胞与其最相似的k个邻居关系
    KNN_list = []

    # 对每个细胞，找出其最相似的k个邻居
    for i in range(similarity_matrix.shape[0]):
        # 根据相似度对所有细胞进行排序（降序）
        sorted_indices = np.argsort(-similarity_matrix[i, :])  # 负号使得排序变为降序
        # 选择最相似的k个细胞（排除自身，因为自身的相似度总是最高的）
        closest_cells = sorted_indices[1:k_neighbors + 1]  
        # 获取对应的相似度值
        closest_distances = similarity_matrix[i, closest_cells]  

        # 创建当前细胞与其k个最相似邻居的连接信息
        KNN_list.append(pd.DataFrame({
            'Cell1': [i] * k_neighbors,  # 当前细胞索引，重复k次
            'Cell2': closest_cells,    # k个最相似的细胞索引
            'Distance': closest_distances  # 相应的相似度值
        }))

    # 将所有细胞的连接信息合并为一个DataFrame
    KNN_df = pd.concat(KNN_list, ignore_index=True)

    # 将细胞数字索引映射回原始的细胞标识符
    id_cell_trans = dict(zip(range(X.shape[0]), X.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

    # 输出网络统计信息（如果verbose=True）
    if verbose:
        print('The graph contains %d edges, %d cells.' % (KNN_df.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (KNN_df.shape[0] / adata.n_obs))

    # 将构建的基因相似度网络保存到adata对象中，便于后续分析
    adata.uns['Gene_Similarity_Net'] = KNN_df

def calculate_ari(adata, ground_truth_key='Ground Truth', cluster_key='mclust'):
    """
    计算调整兰德指数 (ARI)。
    
    Parameters:
    - adata: AnnData object
    - ground_truth_key: str, 真实标签的键名
    - cluster_key: str, 聚类结果的键名
    
    Returns:
    - ARI: float, 调整兰德指数
    """
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df[ground_truth_key], obs_df[cluster_key])
    print(f'Adjusted rand index = {ARI:.4f}')
    return ARI

# utils.txt

# ... (其他函数如 Cal_Spatial_Net, mclust_R, Cal_Gene_Similarity_Net, calculate_ari 等保持不变)

def save_ari_results(all_ari_values, method_name, output_file="ari_results.csv"):
    """
    将单个切片或多个切片的 ARI 值保存到 CSV 文件。
    此版本适用于单切片或批量处理。
    
    Parameters:
    - all_ari_values: dict, 键为切片ID，值为该切片的ARI值。例如: {'151507': 0.54}
    - method_name: str, 当前方法的名称（用于列名）
    - output_file: str, 输出CSV文件的路径
    """
    import pandas as pd
    
    # 将字典转换为DataFrame，切片ID作为索引
    df = pd.DataFrame.from_dict(all_ari_values, orient='index', columns=[method_name])
    df.index.name = 'Section_id'  # 设置索引名称
    
    try:
        # 尝试读取已存在的文件
        existing_df = pd.read_csv(output_file, index_col=0)
        # 合并现有数据和新数据，以Section_id为基准进行合并
        combined_df = existing_df.combine_first(df)
        # 确保新方法的数据被正确填入
        combined_df.update(df)
    except FileNotFoundError:
        # 如果文件不存在，则直接使用当前数据
        combined_df = df
    
    # 保存结果
    combined_df.to_csv(output_file)
    print(f"ARI results saved to {output_file}")
    print(f"Current data: {all_ari_values}")