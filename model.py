import tensorflow.compat.v1 as tf # type: ignore
# 使用TensorFlow 1.x API
tf.disable_v2_behavior()
import warnings

# 忽略所有FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

class GATE():
    """
    GATE (Graph Attention Auto-Encoder) 模型实现
    该模型结合了图注意力机制和自编码器，用于空间转录组学数据分析
    """

    def __init__(self, hidden_dims, alpha=0.8, nonlinear=True, weight_decay=0.0001):
        """
        初始化GATE模型

        参数:
        hidden_dims: 列表，包含每一层的隐藏单元数量，如[1000, 500, 10]表示一个两层的网络
        alpha: 平衡因子，控制原始图和修剪图之间的权重
        nonlinear: 布尔值，是否在隐藏层使用非线性激活函数
        weight_decay: 权重衰减系数，用于L2正则化
        """
        self.n_layers = len(hidden_dims) - 1  # 网络层数
        self.alpha = alpha  # 平衡因子
        self.W, self.v, self.prune_v = self.define_weights(hidden_dims)  # 初始化模型权重
        self.C = {}  # 存储原始图的注意力系数
        self.prune_C = {}  # 存储修剪图的注意力系数
        self.nonlinear = nonlinear  # 是否使用非线性激活
        self.weight_decay = weight_decay  # 权重衰减系数

    def __call__(self, A, prune_A, X):
        """
        模型前向传播

        参数:
        A: 原始邻接矩阵（稀疏张量）
        prune_A: 修剪后的邻接矩阵（稀疏张量）
        X: 输入特征矩阵

        返回:
        loss: 总损失
        H: 节点嵌入表示
        Att_l: 注意力系数
        X_: 重构的输入特征
        """
        # 编码器部分 - 将输入特征编码为低维表示
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, prune_A, H, layer)  # 每层编码操作
            if self.nonlinear:
                print(76543213)  # 调试信息
                if layer != self.n_layers-1:  # 除了最后一层外都使用ELU激活函数
                    H = tf.nn.elu(H)
                    print('1234567')  # 调试信息
        # 最终节点表示
        self.H = H

        # 解码器部分 - 将低维表示解码回原始特征空间
        for layer in range(self.n_layers - 1, -1, -1):  # 反向遍历各层
            H = self.__decoder(H, layer)  # 每层解码操作
            if self.nonlinear:
                if layer != 0:  # 除了第一层外都使用ELU激活函数
                    H = tf.nn.elu(H)
        X_ = H  # 重构的输入特征

        # 特征重构损失 - 计算原始特征和重构特征之间的欧氏距离
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # 权重衰减损失计算
        weight_decay_loss = 0   # 个人感觉应该放到循环外部更好
        for layer in range(self.n_layers):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer]), self.weight_decay, name='weight_loss')

        # 总损失 = 特征重构损失 + 权重衰减损失
        self.loss = features_loss + weight_decay_loss

        # 根据alpha值决定最终的注意力系数输出形式
        if self.alpha == 0:
            self.Att_l = self.C  # 只使用原始图的注意力系数
        else:
            # 下面被注释的代码是另一种合并方式
            # self.Att_l = {x: (1-self.alpha)*self.C[x] + self.alpha*self.prune_C[x] for x in self.C.keys()}
            self.Att_l = {'C': self.C, 'prune_C': self.prune_C}  # 分别返回两种注意力系数
        return self.loss, self.H, self.Att_l, X_

    def __encoder(self, A, prune_A, H, layer):
        """
        编码器的单层实现

        参数:
        A: 原始邻接矩阵
        prune_A: 修剪后的邻接矩阵
        H: 当前层的输入特征
        layer: 当前层的索引

        返回:
        更新后的特征表示
        """
        H = tf.matmul(H, self.W[layer])  # 线性变换
        if layer == self.n_layers-1:  # 最后一层不使用注意力机制
            return H

        # 计算原始图的注意力系数
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)

        if self.alpha == 0:  # 如果alpha为0，只使用原始图
            return tf.sparse_tensor_dense_matmul(self.C[layer], H)
        else:  # 否则结合原始图和修剪图
            # 计算修剪图的注意力系数
            self.prune_C[layer] = self.graph_attention_layer(prune_A, H, self.prune_v[layer], layer)
            # 加权融合两种特征表示
            return (
                    (1-self.alpha) * tf.sparse_tensor_dense_matmul(self.C[layer], H)
                    +
                    self.alpha * tf.sparse_tensor_dense_matmul(self.prune_C[layer], H)
            )

    def __decoder(self, H, layer):
        """
        解码器的单层实现

        参数:
        H: 当前层的输入特征
        layer: 当前层的索引

        返回:
        解码后的特征表示
        """
        H = tf.matmul(H, self.W[layer], transpose_b=True)  # 使用转置权重进行线性变换
        if layer == 0:  # 第一层（实际是解码器的最后一层）不使用注意力机制
            return H

        if self.alpha == 0:  # 如果alpha为0，只使用原始图
            return tf.sparse_tensor_dense_matmul(self.C[layer-1], H)
        else:  # 否则结合原始图和修剪图
            # 加权融合两种特征表示
            return (
                    (1-self.alpha) * tf.sparse_tensor_dense_matmul(self.C[layer-1], H)
                    +
                    self.alpha * tf.sparse_tensor_dense_matmul(self.prune_C[layer-1], H)
            )

    def define_weights(self, hidden_dims):
        """
        定义和初始化模型中的所有权重参数

        参数:
        hidden_dims: 每层的隐藏单元数量列表

        返回:
        W: 权重矩阵字典
        Ws_att: 原始图注意力权重字典
        prune_Ws_att: 修剪图注意力权重字典（如果alpha为0则为None）
        """
        # 初始化每层的权重矩阵
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        # 初始化原始图的注意力权重
        Ws_att = {}
        for i in range(self.n_layers-1):
            v = {}
            # 每层有两个注意力权重向量，用于计算自注意力
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))

            Ws_att[i] = v

        # 如果alpha为0，不需要修剪图的注意力权重
        if self.alpha == 0:
            return W, Ws_att, None

        # 初始化修剪图的注意力权重
        prune_Ws_att = {}
        for i in range(self.n_layers-1):
            prune_v = {}
            prune_v[0] = tf.get_variable("prune_v%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_v[1] = tf.get_variable("prune_v%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_Ws_att[i] = prune_v

        return W, Ws_att, prune_Ws_att

    def graph_attention_layer(self, A, M, v, layer):
        """
        图注意力层的实现

        参数:
        A: 邻接矩阵（稀疏张量）
        M: 节点特征矩阵
        v: 注意力权重参数
        layer: 当前层的索引

        返回:
        attentions: 计算得到的注意力系数（稀疏张量）
        """
        with tf.variable_scope("layer_%s"% layer):
            # 计算注意力分数的第一部分
            f1 = tf.matmul(M, v[0])  # 将特征与第一个注意力向量相乘
            f1 = A * f1  # 只保留有边连接的节点对

            # 计算注意力分数的第二部分
            f2 = tf.matmul(M, v[1])  # 将特征与第二个注意力向量相乘
            f2 = A * tf.transpose(f2, [1, 0])  # 转置并只保留有边连接的节点对

            # 合并两部分得到最终的注意力分数
            logits = tf.sparse_add(f1, f2)

            # 使用sigmoid函数将分数转换为0-1之间的值
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            # 对注意力分数进行softmax归一化
            attentions = tf.sparse_softmax(unnormalized_attentions)

            # 构造最终的注意力系数张量
            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions