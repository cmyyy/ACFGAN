import logging

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tflearn.layers.conv import global_avg_pool
from neuralgym.ops.layers import *
from neuralgym.ops.summary_ops import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

logger = logging.getLogger()
np.random.seed(2018)



def gan_sngan_loss(pos, neg, name='gan_sngan_loss'):
    """
    sngan loss function for GANs.

    - spectral normalization
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(tf.nn.leaky_relu(1.0 - pos)) + tf.reduce_mean(tf.nn.leaky_relu(1.0 + neg))
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss


def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values. Must be one of
      the following types: `float16`, `float32`, `float64`, `int32`, `int64`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    if features.dtype.is_integer:
      features = math_ops.to_float(features)
    alpha = ops.convert_to_tensor(alpha, dtype=features.dtype, name="alpha")
    return math_ops.maximum(alpha * features, features)
##################################################################################
# Spectral Normalization function
##################################################################################

def spectral_norm(w,name ,iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(name+"_u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
# gatedconv+sn
@add_arg_scope
def gated_conv(x, channels, kernel, stride=1, rate=1, use_bias=True, sn=True,
         name='gated_conv', padding='SAME', activation=tf.nn.elu,training=True, return_xg=False):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(kernel-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    xin = x
    if sn and rate ==1:
        w1 = tf.get_variable(name+"_kernel", shape=[kernel, kernel, x.get_shape()[-1], channels])
        w2 = tf.get_variable(name + "_kernel_g", shape=[kernel, kernel, x.get_shape()[-1], channels])
        x = tf.nn.conv2d(input=xin, filter=spectral_norm(w1,name),
                         strides=[1, stride, stride, 1], padding=padding, name=name)
        g = tf.nn.conv2d(input=xin, filter=spectral_norm(w2, name+'_g'),
                         strides=[1, stride, stride, 1], padding=padding, name=name)

        if use_bias :
            bias = tf.get_variable(name+"_bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if  activation:
            x = tf.nn.elu(x, name=name+'_activated')
        g = tf.nn.sigmoid(g, name=name+'_mask_activated')
    elif sn and rate !=1:
        w1 = tf.get_variable(name+"_kernel", shape=[kernel, kernel, x.get_shape()[-1], channels])
        w2 = tf.get_variable(name + "_kernel_g", shape=[kernel, kernel, x.get_shape()[-1], channels])
        x = tf.nn.atrous_conv2d(value=xin, filters=spectral_norm(w1,name),rate = rate,
                     padding=padding, name=name)
        g = tf.nn.atrous_conv2d(value=xin, filters=spectral_norm(w2, name+'_g'),
                         rate = rate, padding=padding, name=name)
        if use_bias :
            bias = tf.get_variable(name+"_bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if activation:
            x = tf.nn.elu(x, name=name+'_activated')
        g = tf.nn.sigmoid(g, name=name+'_mask_activated')
    else :
        x = tf.layers.conv2d(inputs=xin, filters=channels,
                             kernel_size=kernel,strides=stride,
                             dilation_rate=rate, activation=activation, padding=padding,name=name)
        g = tf.layers.conv2d(inputs=xin, filters=channels,
                             kernel_size=kernel, strides=stride,
                             dilation_rate=rate, activation=tf.nn.sigmoid, padding=padding, name=name+'g')
    if not return_xg:
        return tf.multiply(x, g)
    else:
        return x, g
@add_arg_scope
def gated_deconv(x, channels, name='upsample', padding='SAME', sn=True, training=True):
    """Define gated deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gated_conv(
            x, channels, 3, 1, name=name+'_gated_conv', padding=padding,
            training=training, sn=True)
    return x
# CONV+sn
def conv(x, channels, kernel, stride=1, rate=1, use_bias=True, sn=True,
         name='conv', padding='SAME', activation=tf.nn.elu,training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(kernel-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    if sn and rate ==1:
        w = tf.get_variable(name+"_kernel", shape=[kernel, kernel, x.get_shape()[-1], channels])
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w,name),
                         strides=[1, stride, stride, 1], padding=padding, name=name)
        if use_bias :
            bias = tf.get_variable(name+"_bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if activation:
            x = tf.nn.elu(x, name=name+'_activated')
    elif sn and rate !=1:
        w = tf.get_variable(name+"_kernel", shape=[kernel, kernel, x.get_shape()[-1], channels])
        x = tf.nn.atrous_conv2d(value=x, filters=spectral_norm(w,name),rate = rate,
                     padding=padding, name=name)
        if use_bias :
            bias = tf.get_variable(name+"_bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if activation:
            x = tf.nn.elu(x, name=name+'_activated')
    else :
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel,strides=stride,
                             dilation_rate=rate, activation=activation, padding=padding,name=name)
    return x


def deconv(x, cnum, name='upsample', padding='SAME', training=True, sn = True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = conv(
            x, cnum, 3, stride=1, name=name+'_conv', padding=padding,sn=sn,
            training=training)
    return x


##################################################################################
# ASPP
##################################################################################
def atrous_spatial_pyramid_pooling(net, scope, depth=256, reuse=None):
    """
    gated ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
        image_level_features = gated_conv(image_level_features, depth, 1, 1, name="image_level_conv_1x1")

        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = gated_conv(net, depth, 1, 1, name="conv_1x1_0")
        at_pool3x3_1 = gated_conv(net, depth, 3, 1, name="conv_3x3_1", rate=6)
        at_pool3x3_2 = gated_conv(net, depth, 3, 1, name="conv_3x3_2", rate=12)

        at_pool3x3_3 = gated_conv(net, depth, 3, 1, name="conv_3x3_3", rate=18)
        #
        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = gated_conv(net, depth, 1, 1, name='b5_conv2', activation=tf.nn.elu)
        return net


##################################################################################
# attention layer
##################################################################################

def se_gated_attention(x, ch, sn=False, name='se_attention', ratio = 4):
    with tf.variable_scope(name):
        squeeze = global_avg_pool(x, name='Global_avg_pool')
        excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=ch / ratio, name=name + 'fc1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=ch, name=name + 'fc2')
        excitation = tf.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, ch])
        x_scaled = x * excitation
        f = conv(x_scaled, ch // 8, kernel=1, stride=1, sn=sn, name=name+'f') # [bs, h, w, c']

        g = conv(x_scaled, ch // 8, kernel=1, stride=1, sn=sn, name=name+'g') # [bs, h, w, c']

        h = gated_conv(x_scaled, ch, kernel=1, stride=1, sn=sn, name=name+'h') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable(name+"_gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        x = gamma * o + x

    return x


def hw_flatten(x):
    cmy = tf.reshape(x, shape=[x.get_shape().as_list()[0], -1, x.get_shape().as_list()[-1]])
    return cmy
def hwn_flatten(x):
    cmy = tf.reshape(x, shape=[x.get_shape().as_list()[0], x.get_shape().as_list()[1], -1, x.get_shape().as_list()[-1]])
    return cmy

@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True, sn=False, use_bias=True, activation = True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    if sn :
        w = tf.get_variable(name+"_kernel", shape=[ksize, ksize, x.get_shape()[-1], cnum])
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w,name),
                         strides=[1, stride, stride, 1], padding='SAME', name=name)
        if use_bias :
            bias = tf.get_variable(name+"_bias", [cnum], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if activation:
            x = tf.nn.leaky_relu(x, name=name+'_activated')

    else :
        x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
        if activation:
            x = leaky_relu(x)
    return x
def g_dis_conv(x, cnum, ksize=5, stride=2, name='conv', sn=True, use_bias=True, activation = True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        stride: Convolution stride.
        name: Name of layers.

    Returns:
        tf.Tensor: output

    """
    if sn :
        w = tf.get_variable(name+"_kernel", shape=[ksize, ksize, x.get_shape()[-1], cnum])
        x = tf.nn.conv2d(input=x, filter=spectral_norm(w, name),
                         strides=[1, stride, stride, 1], padding='SAME', name=name)

        if use_bias :
            bias = tf.get_variable(name+"_bias", [cnum], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
        if activation:
            x = tf.nn.leaky_relu(x, name=name+'_activated')

    else:
        x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
        if activation:
            x = leaky_relu(x)
    return x

def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
    maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
    t = tf.random_uniform(
        [], minval=config.VERTICAL_MARGIN, maxval=maxt, dtype=tf.int32)
    l = tf.random_uniform(
        [], minval=config.HORIZONTAL_MARGIN, maxval=maxl, dtype=tf.int32)
    h = tf.constant(config.HEIGHT)
    w = tf.constant(config.WIDTH)
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config.SPATIAL_DISCOUNTING_GAMMA
    shape = [1, config.HEIGHT, config.WIDTH, 1]
    if config.DISCOUNTED_MASK:
        logger.info('Use spatial discounting l1 loss.')
        mask_values = np.ones((config.HEIGHT, config.WIDTH))
        for i in range(config.HEIGHT):
            for j in range(config.WIDTH):
                mask_values[i, j] = max(
                    gamma**min(i, config.HEIGHT-i),
                    gamma**min(j, config.WIDTH-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    return tf.constant(mask_values, dtype=tf.float32, shape=shape)

