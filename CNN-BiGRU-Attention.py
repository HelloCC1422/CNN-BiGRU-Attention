# -*- coding: utf-8 -*-
# CNN+双向GRU+attention
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def weight_variables(shape):
    """偏置"""
    w=tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w
  
def bias_variables(shape):
    """偏置"""
    b=tf.Variable(tf.constant(0.001,shape=shape))
    return b
def attention(inputs, attention_size, time_major=False):  
    if isinstance(inputs, tuple):  
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.  
        inputs = tf.concat(inputs, 2)  
  
    if time_major:  
        # (T,B,D) => (B,T,D)  
        inputs = tf.transpose(inputs, [1, 0, 2])  
  
    inputs_shape = inputs.shape  
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer  
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer  
  
    # Attention mechanism  
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  
  
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))  
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))  
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])  
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  
  
    # Output of Bi-RNN is reduced with attention vector  
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)  
  
    return output



def CNN_BIGRU_Attention(x,n_future,n_class,is_training=False):
    x_reshape = tf.reshape(x,[-1,1,n_future,1])
    # 卷积-池化
    with tf.variable_scope("cov1"):
        w_conv1=weight_variables([1,3,1,6]) 
        b_conv1=bias_variables([6])
        #卷积1、BN1、激活1
        x_conv1=tf.nn.conv2d(x_reshape,w_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1
        # 此处应换为一维卷积神经网络
        # x_conv1 = tf.nn.conv1d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1

        # x_conv1=tf.layers.batch_normalization(x_conv1, training=is_training)
        x_relu1=tf.nn.relu(x_conv1)
        #池化1
        x_pool1=tf.nn.max_pool(x_relu1,ksize=[1,1,2,1],strides=[1,1,2,1],padding="SAME")
    # 卷积-池化
    with tf.variable_scope("cov2"):
        w_conv2=weight_variables([1,3,6,16]) 
        b_conv2=bias_variables([16])
        #卷积2、BN2、激活2
        x_conv2=tf.nn.conv2d(x_pool1,w_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2

        # 此处应换为一维神经网络
        # x_conv2 = tf.nn.conv1d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2

        # x_conv2=tf.layers.batch_normalization(x_conv2, training=is_training)
        x_relu2=tf.nn.relu(x_conv2)
        #池化2
        x_pool2=tf.nn.max_pool(x_relu2,ksize=[1,1,2,1],strides=[1,1,2,1],padding="SAME")
    # 拉伸
        pool_shape = x_pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        x_fc_reshape=tf.reshape(x_pool2,[-1,nodes])
    # 全连接层
    with tf.variable_scope("conv_fc"):
        fc1=128
        w_fc1=weight_variables([nodes,fc1])
        b_fc1=bias_variables([fc1])
        y_fc1=tf.matmul(x_fc_reshape,w_fc1)+b_fc1 #未加激活函数
        # y_fc1 =tf.nn.relu(tf.matmul(x_fc_reshape, w_fc1) + b_fc1)
        x_reshape = tf.reshape(y_fc1,[-1,1,fc1])
        #x_reshape=tf.layers.batch_normalization(x_reshape, training=is_training)
    # 双向gru层
    with tf.variable_scope("BIGRU"):
        rnn_cellforword = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(20), tf.nn.rnn_cell.GRUCell(20)])
        rnn_cellbackword = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(20), tf.nn.rnn_cell.GRUCell(20)])
        
        outputs,_=tf.nn.bidirectional_dynamic_rnn(rnn_cellforword,rnn_cellbackword,x_reshape,dtype=tf.float32)
        rnn_out=outputs
        
#        rnn_out = tf.transpose(rnn_out, (1, 0, 2))
    #注意力层
    with tf.variable_scope("attention"):
        attention_size = 64
        attention_out = attention(rnn_out, attention_size, False)
        pool_shape = attention_out.get_shape().as_list()
        nodes = pool_shape[1]
        x_at_reshape=tf.reshape(attention_out,[-1,nodes])
    # 全连接输出层
    with tf.variable_scope("output"):
        w_fc2=weight_variables([nodes,n_class])
        b_fc2=bias_variables([n_class])
        y_predict=tf.matmul(x_at_reshape,w_fc2)+b_fc2 #未加激活函数
        # y_predict =tf.nn.relu(tf.matmul(x_at_reshape, w_fc2) + b_fc2)
        y = tf.reshape(y_predict,[-1,n_class])
    return y











