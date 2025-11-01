# 导入所需的库
import tensorflow.compat.v1 as tf  # type: ignore # 使用TensorFlow 1.x版本的兼容API
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
import scipy.sparse as sp  # 导入稀疏矩阵处理库
import numpy as np  # 导入NumPy进行数值计算
from model import GATE  # 导入自定义的GATE模型
from tqdm import tqdm  # 导入进度条库，用于显示训练进度
import warnings  # 导入警告处理模块

# 忽略所有FutureWarning类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)


class GATES():
    """
    GATES (Graph Attention Encoder) 模型实现类
    用于空间转录组学数据分析的图注意力自编码器
    """

    def __init__(self, hidden_dims, alpha, n_epochs=500, lr=0.0001,
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001,
                 verbose=True, random_seed=2024):
        """
        初始化GATES模型

        参数:
            hidden_dims: 隐藏层维度列表，指定网络结构
            alpha: 平衡参数，控制原始注意力和修剪后注意力的权重
            n_epochs: 训练轮数，默认500
            lr: 学习率，默认0.0001
            gradient_clipping: 梯度裁剪阈值，防止梯度爆炸，默认5
            nonlinear: 是否使用非线性激活函数，默认True
            weight_decay: L2正则化系数，默认0.0001
            verbose: 是否输出训练信息，默认True
            random_seed: 随机种子，用于结果复现，默认2024
        """
        # 设置随机种子，确保结果可复现
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

        # 初始化类属性
        self.loss_list = []  # 存储训练过程中的损失值
        self.lr = lr  # 学习率
        self.n_epochs = n_epochs  # 训练轮数
        self.gradient_clipping = gradient_clipping  # 梯度裁剪阈值
        self.build_placeholders()  # 构建TensorFlow占位符
        self.verbose = verbose  # 是否显示详细训练信息
        self.alpha = alpha  # 平衡原始注意力和修剪注意力的参数

        # 创建GATE模型实例
        self.gate = GATE(hidden_dims, alpha, nonlinear, weight_decay)

        # 运行模型并获取输出（损失、隐藏表示、注意力权重和重构结果）
        self.loss, self.H, self.C, self.ReX = self.gate(self.A, self.prune_A, self.X)

        # 设置优化器
        self.optimize(self.loss)

        # 创建TensorFlow会话
        self.build_session()

    def build_placeholders(self):
        """
        构建模型所需的TensorFlow占位符
        A: 邻接矩阵（稀疏格式）
        prune_A: 经过修剪的邻接矩阵（稀疏格式）
        X: 特征矩阵
        """
        self.A = tf.sparse_placeholder(dtype=tf.float32)  # 原始邻接矩阵，稀疏格式
        self.prune_A = tf.sparse_placeholder(dtype=tf.float32)  # 修剪后的邻接矩阵，稀疏格式
        self.X = tf.placeholder(dtype=tf.float32)  # 节点特征矩阵

    def build_session(self, gpu=True):
        """
        构建TensorFlow会话

        参数:
            gpu: 是否使用GPU，默认为True
        """
        # 设置TensorFlow会话配置
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # GPU显存按需分配

        # 如果不使用GPU，设置线程数为0
        if gpu == False:
            config.intra_op_parallelism_threads = 0  # 操作内并行线程数
            config.inter_op_parallelism_threads = 0  # 操作间并行线程数

        # 创建会话并初始化所有变量
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        """
        设置优化器和梯度裁剪

        参数:
            loss: 要优化的损失函数
        """
        # 使用Adam优化器，设置学习率
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        # 计算梯度
        gradients, variables = zip(*optimizer.compute_gradients(loss))

        # 应用梯度裁剪，防止梯度爆炸
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)

        # 创建训练操作
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, prune_A, X):
        """
        模型调用接口，执行训练过程

        参数:
            A: 原始邻接矩阵（稀疏格式）
            prune_A: 修剪后的邻接矩阵（稀疏格式）
            X: 特征矩阵
        """
        # 循环训练指定轮数，并显示进度条
        for epoch in tqdm(range(self.n_epochs)):
            self.run_epoch(epoch, A, prune_A, X)

    def run_epoch(self, epoch, A, prune_A, X):
        """
        运行一个训练轮次

        参数:
            epoch: 当前轮次
            A: 原始邻接矩阵（稀疏格式）
            prune_A: 修剪后的邻接矩阵（稀疏格式）
            X: 特征矩阵

        返回:
            loss: 本轮的损失值
        """
        # 运行一步训练，计算损失并更新模型参数
        loss, _ = self.session.run([self.loss, self.train_op],
                                   feed_dict={self.A: A,
                                              self.prune_A: prune_A,
                                              self.X: X})

        # 记录损失值
        self.loss_list.append(loss)

        # 如果启用详细输出，打印当前轮次和损失
        if self.verbose:
           print("Epoch: %s, Loss: %.4f" % (epoch, loss))

        return loss

    def infer(self, A, prune_A, X):
        """
        使用训练好的模型进行推理

        参数:
            A: 原始邻接矩阵（稀疏格式）
            prune_A: 修剪后的邻接矩阵（稀疏格式）
            X: 特征矩阵

        返回:
            H: 学习到的节点隐藏表示
            注意力权重: 通过Conbine_Atten_l处理后的注意力权重
            loss_list: 训练过程中的损失记录
            ReX: 重构的特征矩阵
        """
        # 运行模型前向传播，获取隐藏表示、注意力权重和重构结果
        H, C, ReX = self.session.run([self.H, self.C, self.ReX],
                                     feed_dict={self.A: A,
                                                self.prune_A: prune_A,
                                                self.X: X})

        # 返回结果
        return H, self.Conbine_Atten_l(C), self.loss_list, ReX

    def Conbine_Atten_l(self, input):
        """
        处理并合并注意力权重

        参数:
            input: 模型生成的注意力权重

        返回:
            处理后的注意力权重列表（稀疏矩阵格式）
        """
        # 如果alpha为0，直接返回原始注意力
        if self.alpha == 0:
            # 将输入转换为稀疏矩阵格式
            return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                                  shape=(input[layer][2][0], input[layer][2][1])) for layer in input]
        else:
            # 否则，分别处理原始注意力和修剪后的注意力，然后按alpha进行加权组合

            # 转换原始注意力为稀疏矩阵
            Att_C = [sp.coo_matrix((input['C'][layer][1], (input['C'][layer][0][:, 0], input['C'][layer][0][:, 1])),
                                   shape=(input['C'][layer][2][0], input['C'][layer][2][1])) for layer in input['C']]

            # 转换修剪后的注意力为稀疏矩阵
            Att_pruneC = [sp.coo_matrix((input['prune_C'][layer][1], (input['prune_C'][layer][0][:, 0], input['prune_C'][layer][0][:, 1])),
                                        shape=(input['prune_C'][layer][2][0], input['prune_C'][layer][2][1])) for layer in input['prune_C']]

            # 返回加权组合结果：alpha*修剪注意力 + (1-alpha)*原始注意力
            return [self.alpha*Att_pruneC[layer] + (1-self.alpha)*Att_C[layer] for layer in input['C']]
