"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

import inception_resnet_v2
import nips_inception_v3
import nips04_inception_v3
import nips_inception_resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path1', '', 'Path to checkpoint for resnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path3', '', 'Path to checkpoint for nips inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path4', '', 'Path to checkpoint for nips04 inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path5', '', 'Path to checkpoint for nips resnet network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 20, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


class EnsembleModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        return self.run_network(x_input)

    def estimate_y(self, x_input):
        preds = self.run_network(x_input)

        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        return y

    def run_network(self, x_input):
        probs1 = self.run_network1(x_input)
        probs2 = self.run_network2(x_input)
        probs3 = self.run_network3(x_input)
        probs4 = self.run_network4(x_input)
        probs5 = self.run_network5(x_input)

        self.built = True
        return (probs1 + probs2 + probs3 + probs4 + probs5)/5.0

    def run_network1(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

    def run_network2(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

    def run_network3(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nips_inception_v3.inception_v3_arg_scope()):
            _, end_points = nips_inception_v3.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

    def run_network4(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nips04_inception_v3.inception_v3_arg_scope()):
            _, end_points = nips04_inception_v3.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

    def run_network5(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nips_inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = nips_inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    alpha = 2.0 * FLAGS.iter_alpha / 255.0
    num_iter = FLAGS.num_iter

    # Correct alpha according to eps
    alpha = alpha * (FLAGS.max_epsilon / 16.0)

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as g:
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        model = EnsembleModel(num_classes)
        y = model.estimate_y(x_input)

        # 2-step FGSM attack:
        noisy_images = x_input + eps * tf.sign(tf.random_normal(batch_shape))
        x_noisy = tf.clip_by_value(noisy_images, x_min, x_max)

        fgsm = FastGradientMethod(model)

        x_adv = x_noisy
        for _ in range(num_iter):
            # alpha not eps
            #x_next = fgsm.generate(x_adv, y=y, eps=alpha, clip_min=-1., clip_max=1.)
            x_next = fgsm.generate(x_adv, y=y, eps=alpha, clip_min=x_min, clip_max=x_max)
            x_adv = x_next

        # Run computation
        all_vars = tf.all_variables()
        #print(all_vars)

        saver1 = tf.train.Saver([k for k in all_vars if k.name.startswith("InceptionResnetV2")])
        saver2 = tf.train.Saver([k for k in all_vars if (k.name.startswith("Inception") and not k.name.startswith("InceptionResnetV2"))])
        saver3 = tf.train.Saver([k for k in all_vars if k.name.startswith("NipsInceptionV3")])
        saver4 = tf.train.Saver([k for k in all_vars if k.name.startswith("Nips04InceptionV3")])
        saver5 = tf.train.Saver([k for k in all_vars if k.name.startswith("NipsInceptionResnetV2")])

        with tf.Session(graph = g) as sess:
            saver1.restore(sess, FLAGS.checkpoint_path1)
            saver2.restore(sess, FLAGS.checkpoint_path2)
            saver3.restore(sess, FLAGS.checkpoint_path3)
            saver4.restore(sess, FLAGS.checkpoint_path4)
            saver5.restore(sess, FLAGS.checkpoint_path5)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
