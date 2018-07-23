"""A library to train Inception using multiple GPUs with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import time
from datetime import datetime
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# import inception_model as inception
import model as inception
# import mymodel as inception
from config import cfg
from dataset_provider import DatasetProvider

tf.logging.set_verbosity(tf.logging.WARN)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', cfg.PATH_TO_SAVE_MODELS,
                           """Directory where to write event logs and checkpoint.""")

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path',
                           '/DATA/tf_pretrain/inception_v2.ckpt',
                           """If specified, restore this pretrained model before beginning any training.""")


dataset = TSNDataSet(list_file, num_segments=3, new_length=1,
                    modality='RGB', image_tmpl='img_{:05d}.jpg',
                    random_shift=True, test_mode=False)


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
    """Calculate the total loss on a single tower running the ImageNet model.

    We perform 'batch splitting'. This means that we cut up a batch across multiple GPUs.

    Args:
    images: Images. 5D tensor of size [cfg.TRAIN.MINIBATCH, cfg.TRAIN.SEGMENT_NUM,
                                       cfg.TRAIN.IMAGE_HEIGHT, cfg.TRAIN.IMAGE_WIDTH,  cfg.TRAIN.INPUT_CHS].
    labels: 1-D integer Tensor of [cfg.TRAIN.MINIBATCH].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

    Returns:
     Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits = inception.inference(images, num_classes, for_training=True, scope=scope)

    split_batch_size = tf.shape(images)[0]
    inception.loss(logits, labels, batch_size=split_batch_size)
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name +'_raw', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return logits, total_loss


def tower_acc(logits, labels):
    correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), dtype=tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(args):
    """Train on dataset for a number of steps."""

    with tf.Graph().as_default() as g:
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = cfg.TRAIN.EPOCH_SIZE[args.db]['training']//cfg.TRAIN.MINIBATCH
        decay_steps = int(num_batches_per_epoch * cfg.TRAIN.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(cfg.TRAIN.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cfg.TRAIN.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.MomentumOptimizer(lr, momentum=cfg.TRAIN.SGD_MOMENTUM)

        # Get images and labels for ImageNet and split the batch across GPUs.
        assert cfg.TRAIN.MINIBATCH % cfg.TRAIN.NUM_GPUS == 0, (
            'Batch size must be divisible by number of GPUs')

        image_shape = [cfg.TRAIN.MINIBATCH, cfg.TRAIN.SEGMENT_NUM, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.TRAIN.INPUT_CHS]
        label_shape = [cfg.TRAIN.MINIBATCH, ]
        images = tf.placeholder(tf.float32, shape=image_shape, name='images')
        labels = tf.placeholder(tf.int32, shape=label_shape, name='labels')
        images_split = tf.split(value=images, num_or_size_splits=cfg.TRAIN.NUM_GPUS, axis=0)
        labels_split = tf.split(value=labels, num_or_size_splits=cfg.TRAIN.NUM_GPUS, axis=0)

        num_classes = cfg.TRAIN.LABEL_SIZE[args.db]

        # Calculate the gradients for each model tower.
        tower_grads = []
        reuse_variables = None
        logits = []
        for i in range(cfg.TRAIN.NUM_GPUS):
            tf.logging.info('building up graph on gpu{}...'.format(i))
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
                    with slim.arg_scope([slim.variable], device='/cpu:0'):
                        logit, loss = _tower_loss(images_split[i], labels_split[i], num_classes, scope, reuse_variables)
                    reuse_variables = True
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # , 'InceptionV2/Conv2d_2b_1x1/BatchNorm'
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    logits.append(logit[0])

        logits = tf.concat(logits, 0)
        accuracy = tower_acc(logits, labels)

        tf.logging.info('synchronize grads...')
        grads = _average_gradients(tower_grads)

        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        tf.logging.info('add grad summaries...')
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        tf.logging.info('apply grads...')
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        tf.logging.info('add histogram for trainable variables...')
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        tf.logging.info('group all updates into one...')
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
        # train_op = tf.group(apply_gradient_op, variables_averages_op)

        tf.logging.info('create a saver...')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=80)

        summary_op = tf.summary.merge(summaries)

        tf.logging.info('Build an initialization operation to run below...')
        init = tf.global_variables_initializer()

        tf.logging.info('config and init')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        if FLAGS.pretrained_model_checkpoint_path:
            tf.logging.info('load pretrained model...')
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

            variables_to_restore_clean = []
            for v in variables_to_restore:
                if 'Logits' not in v.name:
                    variables_to_restore_clean.append(v)

            restorer = tf.train.Saver(variables_to_restore_clean)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)

            print('#################%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        summary_writer = tf.summary.FileWriter(os.path.join(cfg.PATH_TO_SAVE_MODELS, args.snapshot_pref), graph=sess.graph)

        tf.logging.info('before training...')
        mean = np.array([104, 117, 128])   # BGR
        # mean = np.array([128, 117, 104])
        for step in range(cfg.TRAIN.MAX_STEPS):
            start_time = time.time()
            data_images, data_label = dataset.batch_data()
            image = (data_images / 255.0 - 0.5) * 2.0
            # image = dat['img'] - mean
            feed_dict = {images: image, labels: data_label}
            _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 50 == 0:
                examples_per_sec = cfg.TRAIN.MINIBATCH / float(duration)
                format_str = ('%s: epoch %d, step %d, loss = %.4f, accuracy = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step // int(num_batches_per_epoch), step, loss_value, acc,
                                    examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % num_batches_per_epoch == 0 or (step + 1) == cfg.TRAIN.MAX_STEPS:
                dir_path = os.path.join(FLAGS.train_dir, args.snapshot_pref)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                checkpoint_path = os.path.join(dir_path, '_'.join((args.snapshot_pref, str(step//num_batches_per_epoch), 'model.ckpt')))
                saver.save(sess, checkpoint_path, global_step=step)


def parse_args():
    """ arg parser """
    parser = argparse.ArgumentParser()
    # default xpu0 for non-brain++, all gpus for brain++
    default_devices = '*' if os.environ.get('RLAUNCH_WORKER') else 'xpu0'
    parser.add_argument('-d', '--device', default=default_devices)
    parser.add_argument('--fast-run', action='store_true', default=False)
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    parser.add_argument('-p', '--optimizer',
                        choices=['momentum', 'adam', 'mymomentum'],
                        help='numerical optimizer', default='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--model_name', default=datetime.now().strftime('%Y-%m-%d'),
                        type=str)
    parser.add_argument('--suffix', type=str, default='tf', help="suffix of address")
    parser.add_argument('--db', choices=['act_v1.2', 'act_v1.3', 'ucf', 'kinetics_new', 'kinetics600_standard'],
                        type=str, default='kinetics_new')
    parser.add_argument('--snapshot_pref', type=str)
    parser.add_argument('--init_model', choices=['resnet50', 'resnet152', 'inc_v2_fixed',
                                                 'inc_v2_2xcost', 'waterfall101'],
                        help='select one pretrained model', type=str)
    parser.add_argument('--finetune', dest='finetune_path', required=False)
    parser.add_argument('--partialBN', action='store_true', default=False)
    parser.add_argument('--spatial', action='store_true', default=False)
    parser.add_argument('--temporal', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()
    train(args)


if __name__ == '__main__':
    tf.app.run()
