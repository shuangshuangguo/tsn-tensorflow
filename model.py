from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.framework import ops
from config import cfg
from IPython import embed

FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name to differentiate the operations
TOWER_NAME = 'tower'

BATCHNORM_MOVING_AVERAGE_DECAY = 0.95

MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, for_training=False, scope=None, reuse=None):
    """Build Inception v2 model architecture.
    Args:
      images: Images returned from inputs() or distorted_inputs().
      num_classes: number of classes
      for_training: If set to `True`, build the inference model for training.
        Kernels that operate differently for inference during training
        e.g. dropout, are appropriately configured.
      scope: optional prefix string identifying the ImageNet tower.

    Returns:
      Logits. 2-D float Tensor.
      Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """

    batch_norm_params = {
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001
    }

    # Set weight_decay for weights in Conv and FC layers.
    endpoints = {}
    # seg_num = tf.shape(images)[1]
    # images = tf.reshape(images, [-1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.TRAIN.INPUT_CHS])
    # with variable_scope.variable_scope(scope, 'InceptionV2', [images, num_classes], reuse=reuse) as scope:
    #     with arg_scope([layers_lib.batch_norm], is_training=False):
    #         net, end_points = nets.inception.inception_v2_base(images, scope=scope)
    #         with variable_scope.variable_scope('Logits'):
    #             shape = net.get_shape()
    #             net = layers_lib.avg_pool2d(
    #                 net,
    #                 shape[1:3],
    #                 padding='VALID',
    #                 scope='AvgPool_1a_{}x{}'.format(*shape[1:3]))
    #             # 1 x 1 x 1024
    #             net = layers_lib.dropout(
    #                 net, keep_prob=cfg.TRAIN.DROUPOUT_RATIO, is_training=for_training, scope='Dropout_1b')
    #             logits = slim.conv2d(
    #                 net,
    #                 num_classes, [1, 1],
    #                 activation_fn=None,
    #                 normalizer_fn=None,
    #                 scope='Conv2d_1c_1x1')
    #             logits = tf.reshape(logits, (-1, seg_num, num_classes))
    #             logits = tf.reduce_mean(logits, axis=1)
    #             endpoints['logits'] = logits

    batch_norm_params = {
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'is_training': for_training
    }

    # Set weight_decay for weights in Conv and FC layers.
    with arg_scope([slim.conv2d, slim.fully_connected],
                   weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
                   weights_initializer=slim.xavier_initializer()):
        with arg_scope([slim.conv2d],
                       activation_fn=tf.nn.relu,
                       normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params) as sc:
            # with slim.arg_scope(nets.inception.inception_v2_arg_scope()) as sc:
            seg_num = tf.shape(images)[1]
            images = tf.reshape(images, [-1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.TRAIN.INPUT_CHS])
            logits, _ = nets.inception.inception_v2(images, num_classes=101,
                                               is_training=for_training, dropout_keep_prob=cfg.TRAIN.DROUPOUT_RATIO, reuse=None)
            logits = tf.reshape(logits, (-1, seg_num, num_classes))
            logits = tf.reduce_mean(logits, axis=1)
            endpoints['logits'] = logits

    # with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], trainable=for_training):
    #     with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                         weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
    #                         weights_initializer=slim.xavier_initializer()):
    #         with slim.arg_scope([slim.conv2d],
    #                             activation_fn=tf.nn.relu,
    #                             normalizer_params=batch_norm_params,
    #                             normalizer_fn=slim.batch_norm):
    #
    #             seg_num = tf.shape(images)[1]
    #             images = tf.reshape(images, [-1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.TRAIN.INPUT_CHS])
    #             net, endpoints = nets.inception.inception_v2_base(images, scope='InceptionV2')
    #
    #             with tf.variable_scope('logits'):
    #                 shape = net.get_shape()
    #                 net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='avg_pool')
    #
    #                 # # release depth out
    #                 # shape = net.get_shape()
    #                 # net = tf.reshape(net, tf.stack([-1, seg_num, shape[1], shape[2], shape[3]]))
    #                 # net = tf.reduce_mean(net, axis=1)
    #
    #                 net = slim.dropout(net, keep_prob=cfg.TRAIN.DROUPOUT_RATIO, is_training=for_training, scope='dropout')
    #                 net = slim.flatten(net, scope='flatten')
    #
    #                 logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    #                 logits = tf.reshape(logits, (-1, seg_num, num_classes))
    #                 logits = tf.reduce_mean(logits, axis=1)
    #                 # cls
    #                 endpoints['logits'] = logits
    #                 endpoints['predictions'] = tf.nn.softmax(logits, name='predictions')

    _activation_summaries(endpoints)

    return [endpoints['logits']]


def loss(logits, labels, batch_size=None):
    """Adds all losses for the model.
    Note the final loss is not returned. Instead, the list of losses are collected
    by slim.losses. The losses are accumulated in tower_loss() and summed to
    calculate the total loss.

    Args:
      logits: List of logits from inference(). Each entry is a 2-D float Tensor.
      labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]
      batch_size: integer
    """

    assert batch_size is not None, 'point a value to batch_size'
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    num_classes = logits[0].get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)

    # Cross entropy loss for the main softmax prediction.
    slim.losses.softmax_cross_entropy(logits[0], dense_labels, label_smoothing=0, weights=1.0)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)
