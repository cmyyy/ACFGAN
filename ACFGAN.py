""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import se_gated_attention,gated_conv, gen_conv, gen_deconv,gated_deconv, dis_conv, conv, deconv, gan_sngan_loss, atrous_spatial_pyramid_pooling
from inpaint_ops import random_bbox, bbox2mask, local_patch
from inpaint_ops import spatial_discounting_mask
from inpaint_ops import resize_mask_like


logger = logging.getLogger()


class InpaintModel(Model):
    def __init__(self):
        super().__init__('InpaintModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # Branch convolution
            x = gated_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gated_conv(x, 2 * cnum, 3, 2, name='xconv2_downsample')
            x = gated_conv(x, 2 * cnum, 3, 1, name='xconv3')
            x = gated_conv(x, 4 * cnum, 3, 2, name='xconv4_downsample')
            x = gated_conv(x, 4 * cnum, 3, 1, name='xconv5')
            x = gated_conv(x, 4 * cnum, 3, 1, name='xconv6')
            x = gated_conv(x, 4 * cnum, 3, rate=2, name='xconv7_atrous')
            x = gated_conv(x, 4 * cnum, 3, rate=4, name='xconv8_atrous')
            x = gated_conv(x, 4 * cnum, 3, rate=8, name='xconv9_atrous')
            x = gated_conv(x, 4 * cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # Branch ASPP
            x = gated_conv(xnow, cnum, 5, 1, name='aconv1')
            x = gated_conv(x, 2 * cnum, 3, 2, name='aconv2_downsample')
            x = gated_conv(x, 2 * cnum, 3, 1, name='aconv3')
            x = gated_conv(x, 4 * cnum, 3, 2, name='aconv4_downsample')
            x = gated_conv(x, 4 * cnum, 3, 1, name='aconv5')
            x1 = gated_conv(x, 4 * cnum, 3, 1, name='b1_conv1')
            # IGRB1
            x = gated_conv(x1, 4 * cnum, 3, 1, name='b1_conv2')
            x = gated_conv(x, 4 * cnum, 3, 1, name='b1_conv3', activation=None)
            x = tf.nn.elu(x1 + x, name='r1')
            x2 = gated_conv(x, 4 * cnum, 3, 1, name='b2_conv1')
            # IGRB2
            x = gated_conv(x2, 4 * cnum, 3, 1, name='b2_conv2')
            x = gated_conv(x, 4 * cnum, 3, 1, name='b2_conv3', activation=None)
            x = tf.nn.elu(x2 + x, name='r2')
            x3 = gated_conv(x, 4 * cnum, 3, 1, name='b3_conv1')
            # IGRB3
            x = gated_conv(x3, 4 * cnum, 3, 1, name='b3_conv2')
            x = gated_conv(x, 4 * cnum, 3, 1, name='b3_conv3', activation=None)
            x = tf.nn.elu(x3 + x, name='r3')
            x4 = gated_conv(x, 4 * cnum, 3, 1, name='b4_conv1')
            # IGRB4
            x = gated_conv(x4, 4 * cnum, 3, 1, name='b4_conv2')
            x = gated_conv(x, 4 * cnum, 3, 1, name='b4_conv3', activation=None)
            x = tf.nn.elu(x4 + x, name='r4')
            x5 = gated_conv(x, 4 * cnum, 3, 1, name='b5_conv1')
            # IGRB5
            x = gated_conv(x5, 4 * cnum, 3, 1, name='b5_conv2')
            x = gated_conv(x, 4 * cnum, 3, 1, name='b5_conv3', activation=None)
            x = tf.nn.elu(x5 + x, name='r5')
            x_aspp = atrous_spatial_pyramid_pooling(x, 'ASPP', 4 * cnum)
            # concat all
            x = tf.concat([x_hallu, x_aspp],axis = 3)
            x = se_gated_attention(x, 8 * cnum)
            x = gated_conv(x, 4 * cnum, 3, 1, name='allconv11')
            x = gated_conv(x, 4 * cnum, 3, 1, name='allconv12')
            x = gated_deconv(x, 2 * cnum, name='allconv13_upsample')
            x = gated_conv(x, 2 * cnum, 3, 1, name='allconv14')
            x = gated_deconv(x, cnum, name='allconv15_upsample')
            x = gated_conv(x, cnum // 2, 3, 1, name='allconv16')
            x = conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.tanh(x)
        return x_stage2, offset_flow
    def build_wgan_local_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training, sn=True)
            x = dis_conv(x, cnum*2, name='conv2', training=training, sn=True)
            x = dis_conv(x, cnum*4, name='conv3', training=training, sn=True)
            x = dis_conv(x, cnum*8, name='conv4', training=training, sn=True)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_global', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training,sn=True)
            x = dis_conv(x, cnum*2, name='conv2', training=training,sn=True)
            x = dis_conv(x, cnum*4, name='conv3', training=training,sn=True)
            x = dis_conv(x, cnum*4, name='conv4', training=training,sn=True)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global


    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name='mask_c')
        batch_incomplete = batch_pos*(1.-mask)
        x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training,
            padding=config.PADDING)

        batch_predicted = x2
        logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_batch_predicted = local_patch(batch_predicted, bbox)
        local_patch_x2 = local_patch(x2, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)
        l1_alpha = config.COARSE_L1_ALPHA
		#l1_loss 和 ae_loss本质一样 不过前者计算的是mask里的 后者是外面的
        losses['l1_loss'] = tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))
        losses['ae_loss'] = tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)#这里*2是
			#因为batch_pos_neg此时是原图和修复后的图的堆叠，并且每一张图都需要mask
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss  这里我改成了snganloss
            g_loss_local, d_loss_local = gan_sngan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_sngan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse=True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
            bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name=name+'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)
        batch_predicted = x2
        logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)


    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete
