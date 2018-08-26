# Author : hellcat
# Time   : 18-4-23

import os
import utils
import numpy as np
import tensorflow as tf
from DCGAN_function import dcgan
from WGAN_GP_model import WGAN_GP

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

NUM_FACE = 1 # 生成的脸的数量

XIAOBIANTAI_FACE = [9,3,6,5,4,2,6,4,7,4,
                    4,4,4,3,2,4,6,8,9,4,
                    8,2,4,2,6,3,2,7,4,4,
                    4,3,4,8,2,9,2,9,4,2,
                    4,7,4,4,9,3,6,5,4,2,
                    6,4,3,3,9,4,2,6,5,3,
                    2,4,9,4,9,2,6,4,2,4,
                    2,6,3,2,4,3,9,3,6,5,
                    4,2,6,4,9,6,6,4,9,8,
                    2,6,9,2,4,9,6,7,4,1]
def reload_dcgan():
    if not os.path.exists("./logs/model"):
        tf.logging.info("[*] Failed to find direct './logs/model'")
        return -1
    end_points = dcgan(batch_size=NUM_FACE, train_flag=False)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state("./logs/model")
        if ckpt is not None:
            tf.logging.info("[*] Success to read {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.info("[*] Failed to find a checkpoint")
        sample_z = np.random.uniform(-1, 1, size=(NUM_FACE, 100))

        tmpz = np.array(XIAOBIANTAI_FACE) / 10.0
        tmpz = tmpz.reshape(1, 100)
        sample_z = tmpz

        samples = sess.run(end_points['sample_output'], feed_dict={end_points['initial_z']: sample_z})
        utils.save_images(samples, utils.image_manifold_size(samples.shape[0]),
                          './gcgan_face.png')


def gen_with_wgan():
    if not os.path.exists("./logs/model"):
        tf.logging.info("[*] Failed to find direct './logs/model'")
        return -1

    sess = tf.Session(config=config)

    tmpz = np.array(XIAOBIANTAI_FACE) / 10.0
    tmpz = tmpz.reshape(1, 100)
    sample_z = tmpz

    wgan = WGAN_GP(sess, batch_size=1)
    samples = wgan.train(TRAIN_FLAG=False, batch_z=sample_z)

    utils.save_images(samples, utils.image_manifold_size(samples.shape[0]), './wgan_face.png')
        

if __name__ == '__main__':
    #reload_dcgan()
    gen_with_wgan()
