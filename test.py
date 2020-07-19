import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
import neuralgym as ng
import os
from ACFGAN import InpaintModel
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='pretrained_models/celebahq', type=str,
                    help='The directory of tensorflow checkpoint.')
# ../../model_logs/20190628222355484735_cmy_celebahq256_NORMAL_wgan_gp_full_model0531

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    args = parser.parse_args()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, 256, 256*2, 3))
    output = model.build_server_graph(input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open('data_flist/val.flist', 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    config = ng.Config('inpaint.yml')
    name_int = 0
    for line in lines:
        mask = cv2.imread('../../examples/center_mask_256.png')
        image = cv2.imread(line)

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        path = './results2/'
        if not os.path.exists(path):
            os.makedirs(path)
        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        cv2.imwrite(path + '%s'%str(name_int)+'.png', result[0][:, :, ::-1])
        name_int+=1
    print('Time total: {}'.format(time.time() - t))
