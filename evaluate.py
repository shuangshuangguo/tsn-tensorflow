from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import inception_model as inception
import model as inception
# import mymodel as inception
from config import cfg

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/DATA/tf_output/eval/tsn_inception-v2',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/DATA/tf_output/models/tsn_inception-v2',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

def get_datasets(args, train):
    if train:
        dataset=DatasetProvider(dataset_name=args.db,
                            phase='training',
                            suffix=args.suffix)
    else:
        dataset=DatasetProvider(dataset_name=args.db,
                            phase='validation',
                            suffix=args.suffix)

    from dpflow import Controller
    controller = Controller(io=[dataset._receiver])
    controller.start()

    return dataset


def _eval_once(args, saver, top_1_op, top_5_op, images, labels, model_path):
    """Runs Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt.model_checkpoint_path))

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        num_iter = int(math.ceil(cfg.TRAIN.EPOCH_SIZE[args.db]['validation'] // cfg.TEST.MINIBATCH))
        count_top_1 = 0.0
        count_top_5 = 0.0
        total_sample_count = num_iter * cfg.TEST.MINIBATCH
        step = 0

        print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
        start_time = time.time()

        # Read data from dpflow
        def get_inf_iter_from_dataset(ds):
            def get_inf_iter_ds():
                while True:
                    yield from ds.get_epoch_minibatch_iter()

            return iter(get_inf_iter_ds())

        datasets = get_datasets(args, train=False)
        ds_iter = get_inf_iter_from_dataset(datasets)

        # PreLoad Data
        for i in range(5):
            dat = next(ds_iter)

        acc_list = []
        while step < num_iter:
            dat = next(ds_iter)
            image = (dat['img'] / 255.0 - 0.5) * 2.0
            feed_dict = {images: image, labels: dat['label']}
            top_1, top_5 = sess.run([top_1_op, top_5_op], feed_dict=feed_dict)
            count_top_1 += np.sum(top_1)
            count_top_5 += np.sum(top_5)
            step += 1
            acc = np.sum(top_1) / cfg.TEST.MINIBATCH
            acc_list.append(acc)
            if step % 200 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / 20.0
                examples_per_sec = cfg.TEST.MINIBATCH / sec_per_batch
                print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                    'sec/batch), accuracy: %f' % (datetime.now(), step, num_iter,
                                    examples_per_sec, sec_per_batch, acc))
                start_time = time.time()

        # Compute precision @ 1.
        precision_at_1 = count_top_1 / total_sample_count
        recall_at_5 = count_top_5 / total_sample_count
        print(count_top_1, total_sample_count)
        print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
              (datetime.now(), precision_at_1, recall_at_5, total_sample_count))
        print(model_path, 'accuracy: ', np.mean(np.array(acc_list)))

        return precision_at_1


def evaluate(args):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        image_shape = [cfg.TEST.MINIBATCH, cfg.TEST.SEGMENT_NUM, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH,
                       cfg.TRAIN.INPUT_CHS]
        label_shape = [cfg.TEST.MINIBATCH, ]
        images = tf.placeholder(tf.float32, shape=image_shape, name='images')
        labels = tf.placeholder(tf.int32, shape=label_shape, name='labels')

        num_classes = cfg.TRAIN.LABEL_SIZE[args.db]

        logits = inception.inference(images, num_classes, for_training=False)[0]

        # Calculate predictions.
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)

        while True:
            _eval_once(args, saver, summary_writer, top_1_op, top_5_op, images, labels)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', choices=['ucf', 'kinetics', 'kinetics600'],
                        type=str, default='ucf')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    tf.app.run()
