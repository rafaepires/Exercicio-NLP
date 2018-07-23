import os
import numpy as np
import tensorflow as tf
import math

def save(session, file_path):
    saver = tf.train.Saver()
    saver.save(session, file_path)
    return


def generateBatches(data, label, step, batchsize, last_batch = False):
    if not last_batch:
        batchdata = data[step * batchsize : (step + 1) * batchsize, :]
        batchlabel = label[step * batchsize : (step + 1) * batchsize, :]
    else:
        batchdata = data[step * batchsize :, :]
        batchlabel = label[step * batchsize :, :]
    return (batchdata, batchlabel)


def one_hot(labels, num_classes):
    hot_labels = np.zeros((labels.shape[0],num_classes))
    for raw in range( hot_labels.shape[0] ):
        coluna = labels[int(raw),(0)]
        hot_labels[ int(raw) , int(coluna)] = 1
    return hot_labels


def linear(in_layer, output_dim, scope=None):

    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [in_layer.get_shape()[1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer(),
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(in_layer, w) + b


def build_layer(in_layer, h_dim, act_type='', layer_name=''):
    if act_type == 'sigmoid':
        h0 = tf.nn.sigmoid(linear(in_layer, h_dim, layer_name))
    elif act_type == 'softmax':
        h0 = tf.nn.softmax(linear(in_layer, h_dim, layer_name))
    elif act_type == 'relu':
        h0 = tf.nn.relu(linear(in_layer, h_dim, layer_name))
    elif act_type == 'tanh':
        h0 = tf.nn.tanh(linear(in_layer, h_dim, layer_name))
    else:
        h0 = linear(in_layer, h_dim, layer_name)
    return h0


def train(data, label, batch_sz, epochs, num_classes, model_path):

    hot_labels = one_hot(label,num_classes)
    idx_perm = np.random.permutation(data.shape[0])
    data = data[idx_perm, :]
    hot_labels = hot_labels[idx_perm, :]
    
    x = tf.placeholder(tf.float32, [None, data.shape[1]])

    with tf.variable_scope('mlp'):
        layer1 = build_layer(x, 1000, act_type='relu', layer_name='h1')
        layer2 = build_layer(layer1, 1000, act_type='relu', layer_name='h2')
        layer3 = build_layer(layer2, 1000, act_type='relu', layer_name='h3')
        y = build_layer(layer3, num_classes, act_type='', layer_name='out')

        y_ = tf.placeholder(tf.float32, [None, num_classes])

    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        J_history = []
        sess.run(init)
        num_steps = math.ceil(data.shape[0] / batch_sz)

        for epoch in range(epochs):

            last_batch = False
            for step in range(num_steps):
                if step == num_steps - 1:
                    last_batch = True

                batch_data, batch_label = generateBatches(data, hot_labels, step, batch_sz, last_batch)
              
                feed = {x:batch_data, y_:batch_label}
                loss_value, _ = sess.run([cross_entropy, train_step], feed_dict = feed)
                print(loss_value)
    
        os.system('mkdir -p ' + model_path)
        save(sess, model_path + '/save_files')


def predict(data, num_classes, model_path):

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, data.shape[1]])

    with tf.variable_scope('mlp'):
        layer1 = build_layer(x, 1000, act_type='relu', layer_name='h1')
        layer2 = build_layer(layer1, 1000, act_type='relu', layer_name='h2')
        layer3 = build_layer(layer2, 1000, act_type='relu', layer_name='h3')
        y = build_layer(layer3, num_classes, act_type='', layer_name='out')
        prediction = tf.argmax(tf.nn.softmax(y), 1)


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path + '/save_files')
        predictions = sess.run(prediction, feed_dict = {x:data})

    return predictions
   
