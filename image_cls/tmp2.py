# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--A_dir", help="path to folder containing A images")
parser.add_argument("--B_dir", help="path to folder containing B images")
parser.add_argument("--D_dir", help="path to folder containing D images")

parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_frelina1"
                         "q steps")
parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")





# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths_A, paths_B,paths_D ,label_B,a_image,b_image,d_images, count, steps_per_epoch")

Model = collections.namedtuple("Model", "g_loss_a2b, g_loss_b2a, g_loss,da_loss,g_loss_water "
                                        " db_loss1,db_loss2,db_loss_fake2,db_loss_real2,discrim_grads_and_vars, gen_grads_and_vars, outputs_a, output_b, train,label, fc8r")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb
def lrelu1(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def load_examples():

    #A_paths = glob.glob(os.path.join(a.A_dir, "*.jpg"))
    A_paths = glob.glob(os.path.join(a.A_dir, "*.jpg"))
    B_paths = glob.glob(os.path.join(a.B_dir, "*.jpg"))

    D_paths = glob.glob(os.path.join(a.D_dir, '*.jpg'))
    decode_A = tf.image.decode_jpeg
    if len(A_paths) == 0:
        A_paths = glob.glob(os.path.join(a.A_dir, "*.png"))
        decode_A = tf.image.decode_png

    if len(A_paths) == 0:
        raise Exception("input_dir contains no image files")


    decode_D = tf.image.decode_jpeg
    if len(D_paths) == 0:
        D_paths = glob.glob(os.path.join(a.D_dir, "*.png"))
        decode_D = tf.image.decode_png

    if len(D_paths) == 0:
        raise Exception("input_dir contains no image files")





    decode_B = tf.image.decode_jpeg
    if len(B_paths) == 0:
        B_paths = glob.glob(os.path.join(a.B_dir, "*.png"))
        decode_B = tf.image.decode_png

    if len(B_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in A_paths) and all(get_name(path_B).isdigit() for path_B in B_paths) and all(get_name(path_D).isdigit() for path_D in D_paths):
        A_paths = sorted(A_paths, key=lambda path: int(get_name(path)))
        B_paths = sorted(B_paths, key=lambda path_B: int(get_name(path_B)))
        D_paths = sorted(D_paths, key=lambda path_D: int(get_name(path_D)))
    else:
        # random.shuffle(A_paths)
        # random.shuffle(B_paths)
        A_paths = sorted(A_paths)
        B_paths = sorted(B_paths)
        D_paths = sorted(D_paths)
    img_name_label_B = np.zeros((len(B_paths), 1))
    img_name_label_D = np.zeros((len(D_paths), 1))
    for i in range(len(B_paths)):
        name = B_paths[i].split('/')
        name_img = name[2]
        # name_img = name[7]
        nn = name_img.split('.')
        img_name = nn[0]
        label = int(img_name[-1])
        img_name_label_B[i][0] = label
    for i in range(len(D_paths)):
        name = D_paths[i].split('/')
        name_img = name[2]
        # name_img = name[7]
        nn = name_img.split('.')
        img_name = nn[0]
        label = int(img_name[-1])
        img_name_label_D[i][0] = label

    with open('fen000.csv', "w") as foo:
        np.savetxt(foo, img_name_label_B, delimiter=',')
    with open('fend000.csv', "w") as foo:
        np.savetxt(foo, img_name_label_D, delimiter=',')
    with tf.name_scope("load_images_A"):
        path_queue = tf.train.string_input_producer(A_paths, shuffle=False)
        # tf.train.string_input_producer
        reader = tf.WholeFileReader()
        paths_A, contents = reader.read(path_queue)
        raw_input = decode_A(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])
        a_images = preprocess(raw_input)

    with tf.name_scope("load_images_B"):
        path_queue = tf.train.string_input_producer(B_paths, shuffle=False)

        label_queue = tf.train.string_input_producer(['fen000.csv'], shuffle=False)
        reader_label = tf.TextLineReader()
        reader = tf.WholeFileReader()
        key, val = reader_label.read(label_queue)
        record_defaults = [[]]
        img_l = tf.decode_csv(val, record_defaults=record_defaults)
        # str_img=tf.cast(img_l,tf.string)
        label_tensor = tf.cast(img_l, tf.int32)
        img_label_B = tf.one_hot(label_tensor, 7)
        paths_B, contents = reader.read(path_queue)
        raw_input = decode_B(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])
        b_images = preprocess(raw_input)

    with tf.name_scope("load_images_D"):
        path_queue = tf.train.string_input_producer(D_paths, shuffle=False)

        label_queue = tf.train.string_input_producer(['fend000.csv'], shuffle=False)
        reader_label = tf.TextLineReader()
        reader = tf.WholeFileReader()
        key, val = reader_label.read(label_queue)
        record_defaults = [[]]
        img_l = tf.decode_csv(val, record_defaults=record_defaults)
        # str_img=tf.cast(img_l,tf.string)
        label_tensor = tf.cast(img_l, tf.int32)
        img_label_D = tf.one_hot(label_tensor, 7)
        paths_D, contents = reader.read(path_queue)
        raw_input = decode_D(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])
        d_images = preprocess(raw_input)

    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("A_images"):
        a_images = transform(a_images)

    with tf.name_scope("target_images"):
        b_images = transform(b_images)

    with tf.name_scope("target_imagesD"):
        d_images = transform(d_images)

    img_label_B = tf.reshape(img_label_B, [1, 7, -1])
    img_label_D = tf.reshape(img_label_D, [1, 7, -1])
    paths_batch_A,paths_batch_B, paths_batch_D,a_images_batch, b_images_batch,d_images_batch,img_label_B_batch ,img_label_D_batch= tf.train.batch(
        [paths_A,paths_B ,paths_D,a_images, b_images,d_images,img_label_B,img_label_D], batch_size=a.batch_size)

    steps_per_epoch = int(math.ceil(max(len(B_paths),len(A_paths)) / a.batch_size))

    return Examples(
        paths_A=paths_batch_A,#A类图的路径
        paths_B=paths_batch_B,#B类图的路径
        paths_D=paths_batch_D,
        label_B=img_label_B_batch,
        a_image=a_images_batch,
        b_image=b_images_batch,
        d_images=d_images_batch,
        count=max(len(A_paths),len(B_paths)),
        steps_per_epoch=steps_per_epoch,

    )


#def batch_norm(x, name="batch_norm"):
#    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)

def ssim(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 13, 2, 'VALID')
        mu_y = slim.avg_pool2d(y, 13, 2, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 13, 2, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 13, 2, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 13, 2, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def L1(a,b):
    little_a = slim.avg_pool2d(a,13, 2, "VALID")
    little_b = slim.avg_pool2d(b,13, 2, "VALID")
    l1= tf.reduce_mean(tf.abs(little_a-little_b))
    return l1


def generator_water( image, depth,num, auxiliary,reuse=False, name="generator_water"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with tf.variable_scope("generator_water") as scope:
            output_height = 256
            output_width = 256
            batch_size = 1

            g_bn0 = batch_norm(name='g_bn0')
            g_bn1 = batch_norm(name='g_bn1')
            g_bn2 = batch_norm(name='g_bn2')
            g_bn3 = batch_norm(name='g_bn3')
            g_bn4 = batch_norm(name='g_bn4')

            r2 = np.ones([output_height, output_width], np.float32)
            r4 = np.ones([output_height, output_width], np.float32)
            r6 = np.ones([output_height, output_width], np.float32)
            cx = output_width / 2
            cy = output_height / 2
            for i in range(0, output_height):
                for j in range(0, output_width):
                    r = np.sqrt((i - cy) * (i - cy) + (j - cx) * (j - cx)) / (np.sqrt(cy * cy + cx * cx))
                    r2[i, j] = r * r
                    r4[i, j] = r * r * r * r
                    r6[i, j] = r * r * r * r * r * r
          # water-based attenuation and backscatter

            with tf.variable_scope("fc"):

                fc = tf.get_variable("full-connect", shape=[1, 1, 256 * 256 * 3, num], dtype=tf.float32,
                                     initializer=tf.ones_initializer())
                auxiliary = tf.matmul(fc, auxiliary)
            r = tf.reshape(auxiliary, [1, 256, 256, 3])
            image += r
            with tf.variable_scope("g_atten"):
              #第一阶段公式,注意初始化的参数和之后将其进行打包
                init_r = tf.random_normal([1,1,1],mean=0.35,stddev=0.01,dtype=tf.float32)
                eta_r = tf.get_variable("g_eta_r",initializer=init_r)
                init_b = tf.random_normal([1,1,1],mean=0.015,stddev=0.01,dtype=tf.float32)
                eta_b = tf.get_variable("g_eta_b",initializer=init_b)
                init_g = tf.random_normal([1,1,1],mean=0.036,stddev=0.01,dtype=tf.float32)
                eta_g = tf.get_variable("g_eta_g",initializer=init_g)
                eta = tf.stack([eta_r,eta_g,eta_b],axis=3)
                eta_d = tf.exp(tf.multiply(-1.0,tf.multiply(depth,eta)))#depth就是rc

            h0 = tf.multiply(image,eta_d)#这里传进去的image是Iair,I air e −η(λ
          #返回的参数是G1
         #    z =
         # # backscattering
         #    z_,h0z_w, h0z_b = linear(z,output_width*output_height*batch_size*1, 'g_h0_lin', with_w=True)
         #  #把参数也进行传回了
         #    h0z = tf.reshape(z_, [-1, output_height, output_width, 1])
         #    h0z = tf.nn.relu(g_bn0(h0z))
         #    h0z = tf.multiply(h0z,depth)
         #    print(h0z.get_shape)

            #copy 3份，分别作为rgb的输入
            with tf.variable_scope('g_h1_conv'):
                 w = tf.get_variable('g_w',[ 5,5, h0.get_shape()[-1], 1],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1z = tf.nn.conv2d(h0, w, strides=[1, 1,1, 1], padding='SAME')
            h_g = lrelu1(g_bn1(h1z))

            with tf.variable_scope('g_h1_convr'):
                wr = tf.get_variable('g_wr',[ 5,5, h0.get_shape()[-1], 1],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zr = tf.nn.conv2d(h0, wr, strides=[1, 1, 1, 1], padding='SAME')
            h_r = lrelu1(g_bn3(h1zr))

            with tf.variable_scope('g_h1_convb'):
                wb = tf.get_variable('g_wb',[ 5,5, h0.get_shape()[-1], 1],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1zb = tf.nn.conv2d(h0, wb, strides=[1, 1, 1, 1], padding='SAME')
            h_b = lrelu1(g_bn4(h1zb))#进行batch norm的操作
               #去掉最后一层维度为1
            h_r = tf.squeeze(h_r,axis=3)
            h_g = tf.squeeze(h_g,axis=3)
            h_b = tf.squeeze(h_b,axis=3)

            h_final=tf.stack([h_r,h_g,h_b],axis=3)
              #将三个参数进行连接，然后就变成了一个RGB的最终生成图

            h2 = tf.add(h_final,h0)#拿第一阶段的输出与第二阶段的输出直接矩阵相加

              # camera model
            with tf.variable_scope("g_vig"):
                  A = tf.get_variable('g_amp', [1],
                      initializer=tf.truncated_normal_initializer(mean=0.9,stddev=0.01))
                  C1 = tf.get_variable('g_c1', [1],
                      initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
                  C2 = tf.get_variable('g_c2', [1],
                      initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
                  C3 = tf.get_variable('g_c3', [1],
                      initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.001))
              #三通道的R2,R4,R6和一个参数进行相除
            h11 = tf.multiply(r2,C1)
            h22 = tf.multiply(r4,C2)
            h33 = tf.multiply(r6,C3)
            h44 = tf.ones([output_height,output_width],tf.float32)
            h1 = tf.add(tf.add(h44,h11),tf.add(h22,h33))#将四个通道的值进行相加，然后直接相除
            V = tf.expand_dims(h1,axis=2)
            h1a = tf.divide(h2,V)
            h_out = tf.multiply(h1a,A)
            return h_out, eta_r,eta_g,eta_b, C1,C2,C3,A


def generator_resnet(image, num, auxiliary, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        with tf.variable_scope("fc"):
            # fc = tf.get_variable("full-connect1", shape=[1, 1, 256*256*3, num], dtype=tf.float32,
            #                      initializer=tf.random_normal_initializer(1.0, 0.02))
            fc = tf.get_variable("full-connect", shape=[1, 1, 256 * 256 * 3, num], dtype=tf.float32,
                                 initializer=tf.ones_initializer())
            # bias = tf.get_variable("bias", shape=[1, 1,1, num], dtype=tf.float32,
            #                         initializer=tf.random_normal_initializer(1.0, 0.02))
            # auxiliary [1,1,num,1]
            auxiliary = tf.matmul(fc, auxiliary)
        r = tf.reshape(auxiliary, [1, 256, 256, 3])
        image += r
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, a.ngf, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, a.ngf * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, a.ngf * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, a.ngf * 4, name='g_r1')
        r2 = residule_block(r1, a.ngf * 4, name='g_r2')
        r3 = residule_block(r2, a.ngf * 4, name='g_r3')
        r4 = residule_block(r3, a.ngf * 4, name='g_r4')
        r5 = residule_block(r4, a.ngf * 4, name='g_r5')
        r6 = residule_block(r5, a.ngf * 4, name='g_r6')
        r7 = residule_block(r6, a.ngf * 4, name='g_r7')
        r8 = residule_block(r7, a.ngf * 4, name='g_r8')
        r9 = residule_block(r8, a.ngf * 4, name='g_r9')

        #with tf.variable_scope("fc"):
            # fc = tf.get_variable("full-connect", shape=[1, 1, a.ngf * a.ngf * a.ngf * 4, num], dtype=tf.float32,
            #                      initializer=tf.random_normal_initializer(1.0, 0.02))
            #fc = tf.get_variable("full-connect", shape=[1, 1, a.ngf * a.ngf * a.ngf * 4, num], dtype=tf.float32,
        #                         initializer=tf.ones_initializer())
            # bias = tf.get_variable("bias", shape=[1, 1,1, num], dtype=tf.float32,
            #                         initializer=tf.random_normal_initializer(1.0, 0.02))
            # auxiliary [1,1,num,1]
            #auxiliary = tf.matmul(fc, auxiliary)
        #r = tf.reshape(auxiliary, [1, 64, 64, 64 * 4])
        #r9 += r
        d1 = deconv2d(r9, a.ngf * 2, 3, 2, name='g_d1_dc')

        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, a.ngf, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def create_model(a_images, b_images,label_B,d_images):
    def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1):
        """Create a convolution layer.

        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels / groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu
    def fc(x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                                      trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

    def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def lrn(x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def dropout(x, keep_prob):
        """Create a dropout layer."""
        return tf.nn.dropout(x, keep_prob)

    def discriminator1(image,num,label,reuse=False,name="discriminator1"):
        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False


            with tf.variable_scope("fc"):
                # fc = tf.get_variable("full-connect1", shape=[1, 1, 256*256*3, num], dtype=tf.float32,
                #                      initializer=tf.random_normal_initializer(1.0, 0.02))
                fc = tf.get_variable("full-connect1", shape=[1, 1, 256*256*3, num], dtype=tf.float32,
                                     initializer=tf.ones_initializer())
                # bias = tf.get_variable("bias", shape=[1, 1,1, num], dtype=tf.float32,
                #                         initializer=tf.random_normal_initializer(1.0, 0.02))
                # auxiliary [1,1,num,1]
                auxiliary = tf.matmul(fc, label)
            r = tf.reshape(auxiliary, [1, 256, 256, 3])
            image+=r
            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'),0.2)
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, 64 * 2, name='d_h1_conv'), 'd_bn1'),0.2)
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, 64 * 4, name='d_h2_conv'), 'd_bn2'),0.2)
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, 64 * 8, s=1, name='d_h3_conv'), 'd_bn3'),0.2)
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
            # h4 is (32 x 32 x 1)

            h4sg=tf.sigmoid(h4)

            return  h4sg
    def discriminator2(image,num,label,reuse=False,name="discriminator2"):
        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            """Create the network graph."""
            # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
            image = tf.image.resize_images(image, [227, 227])
            conv1 = conv(image, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
            pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

            # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
            conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

            # 3rd Layer: Conv (w ReLu)
            conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
            fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
            dropout6 = dropout(fc6, 0.5)

            # 7th Layer: FC (w ReLu) -> Dropout
            fc7 = fc(dropout6, 4096, 4096, name='fc7')
            dropout7 = dropout(fc7, 0.5)

            # 8th Layer: FC and return unscaled activations
            fc8 = fc(dropout7, 4096, num, relu=False, name='fc8')
            fc8r = tf.reshape(fc8,[1,1,num,1])
            # sess = tf.Session()
            # print(sess.run(fc8r))
            # print(sess.run(label))
            with tf.name_scope("cross_ent"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=fc8r))

            return  loss, label, fc8r

    with tf.variable_scope("generatorA2B"):
        #alpha=tf.Constant([1,256,256,3],0.8)
        alpha = tf.Variable(tf.constant(0.9,dtype=tf.float32,shape=[1,256,256,3]))
        tmp1=generator_resnet(a_images, 7, label_B, False, "generator_resnet_A2B")
        tmp2,eta_r_real,eta_g_real,eta_b_real,C1_real,C2_real,C3_real,A_real=generator_water(a_images,d_images,7,label_B,False,"generator_water_A2B")
        fake_b = tf.multiply(tmp1,alpha)+tf.multiply(tmp2,1-alpha)

    with tf.variable_scope("generatorB2A"):
        recover_a=generator_resnet(fake_b,7,label_B,False,"generator_resnet_B2A")


    with tf.variable_scope("generatorB2A",reuse=True):
        fake_a=generator_resnet(b_images,7,label_B,True,"generator_resnet_B2A")

    with tf.variable_scope("generatorA2B",reuse=True):
        recover_b=generator_resnet(fake_a,7,label_B,True,"generator_resnet_A2B")


    DB_fake=discriminator1(fake_b,num=7,label=label_B,reuse=False,name="discriminatorB1")
    DA_fake=discriminator1(fake_a,num=7,label=label_B,reuse=False,name="discriminatorA1")
    DB_real=discriminator1(b_images,num=7,label=label_B,reuse=True,name="discriminatorB1")
    DA_real=discriminator1(a_images,num=7,label=label_B,reuse=True,name="discriminatorA1")

    DBS_fake, label1, fc8rs= discriminator2(fake_b, 7, label_B, reuse=False, name="discriminatorB2")
    #DAS_fake= discriminator1(fake_a, 7, label_B, reuse=False, name="discriminatorA1")
    DBS_real,label2, fc8rs1= discriminator2(b_images, 7, label_B, reuse=True, name="discriminatorB2")
    #DAS_real= discriminator1(a_images, 7, label_B, reuse=True, name="discriminatorA1")

    with tf.name_scope("generator_loss"):
        g_loss_a2b=tf.reduce_mean((DB_fake-tf.ones_like(DB_fake))**2)+tf.reduce_mean((DBS_fake-tf.zeros_like(DBS_fake,dtype=tf.float32))**2)*1.0+tf.reduce_mean(tf.abs(a_images-recover_a))*10+tf.reduce_mean(tf.abs(b_images - recover_b)) * 10+L1(fake_b,b_images)*2.0+tf.reduce_mean(ssim(a_images,fake_b))*1.0
        g_loss_b2a = tf.reduce_mean((DA_fake-tf.ones_like(DA_fake))** 2)+tf.reduce_mean(tf.abs(b_images - recover_b)) * 10+tf.reduce_mean(tf.abs(a_images-recover_a))*10+L1(fake_a,a_images)*2.0+tf.reduce_mean(ssim(b_images,fake_a))*1.0
        g_loss=tf.reduce_mean((DB_fake-tf.ones_like(DB_fake) )** 2)+tf.reduce_mean((DA_fake-tf.ones_like(DA_fake) )** 2)+tf.reduce_mean((DBS_fake-tf.zeros_like(DBS_fake,dtype=tf.float32))**2) + tf.reduce_mean(tf.abs(a_images - recover_a)) * 10 +\
               tf.reduce_mean(tf.abs(b_images - recover_b)) * 10+L1(fake_a,a_images)*2.0+L1(fake_b,b_images)*2.0+tf.reduce_mean(ssim(a_images,fake_b))*1.0+tf.reduce_mean(ssim(b_images,fake_a))*1.0
        # g_loss_a2b=tf.reduce_mean((DB_fake-tf.ones_like(DB_fake))**2)+tf.reduce_mean((DBS_fake-tf.zeros_like(DBS_fake,dtype=tf.float32))**2)+tf.reduce_mean(tf.abs(a_images-recover_a))*10.0+tf.reduce_mean(tf.abs(b_images - recover_b)) * 10.0+tf.reduce_mean(tf.abs(fake_b-b_images))*0.5+tf.reduce_mean(ssim(a_images,fake_b))*0.5
        # g_loss_b2a = tf.reduce_mean((DA_fake-tf.ones_like(DA_fake))** 2)+ tf.reduce_mean((DAS_fake-tf.zeros_like(DAS_fake,dtype=tf.float32))**2)+tf.reduce_mean(tf.abs(b_images - recover_b)) * 10.0+tf.reduce_mean(tf.abs(a_images-recover_a))*10.0+tf.reduce_mean(tf.abs(fake_a-a_images))*0.5+tf.reduce_mean(ssim(b_images,fake_a))*0.5
        # g_loss=tf.reduce_mean((DB_fake-tf.ones_like(DB_fake) )** 2)+tf.reduce_mean((DA_fake-tf.ones_like(DA_fake) )** 2)+tf.reduce_mean((DBS_fake-tf.zeros_like(DBS_fake,dtype=tf.float32))**2) +tf.reduce_mean((DAS_fake-tf.zeros_like(DAS_fake,dtype=tf.float32))**2)+ tf.reduce_mean(tf.abs(a_images - recover_a)) * 10.0 +\
        #        tf.reduce_mean(tf.abs(b_images - recover_b)) * 10.0+tf.reduce_mean(tf.abs(fake_b-b_images))*0.5+tf.reduce_mean(tf.abs(fake_a-a_images))*0.5+tf.reduce_mean(ssim(a_images,fake_b))*0.5+tf.reduce_mean(ssim(b_images,fake_a))*0.5
        c1_loss=-tf.minimum(tf.reduce_min(C1_real),0)*10000
        c2_loss=-tf.minimum(tf.reduce_min(-1*(4*C2_real*C2_real-12*C1_real*C3_real)),0)*10000
        eta_r_loss = -tf.minimum(tf.reduce_min(eta_r_real), 0) * 10000
        eta_g_loss = -tf.minimum(tf.reduce_min(eta_g_real), 0) * 10000
        eta_b_loss = -tf.minimum(tf.reduce_min(eta_b_real), 0) * 10000
        A_loss = -tf.minimum(tf.reduce_min(A_real),0)*10000
        g_loss_water = c1_loss + c2_loss + tf.reduce_mean((DB_fake-tf.ones_like(DB_fake))**2)+ eta_r_loss + eta_g_loss +eta_b_loss + A_loss

    with tf.name_scope("discriminator_loss"):

        db_loss_real1=tf.reduce_mean((DB_real-tf.ones_like(DB_real))**2)
        db_loss_fake1 = tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake)) ** 2)
        db_loss1=(db_loss_real1 + db_loss_fake1)/2.0

        db_loss_real2=tf.reduce_mean((DBS_real-tf.zeros_like(DBS_real,dtype=tf.float32))**2)
        db_loss_fake2 = tf.reduce_mean((DBS_fake-tf.zeros_like(DBS_fake,dtype=tf.float32))**2)

        db_loss2 = (db_loss_real2 + db_loss_fake2) / 2.0

        db_loss=db_loss1+db_loss2


        da_loss_real1 = tf.reduce_mean((DA_real - tf.ones_like(DA_real)) ** 2)
        da_loss_fake1 = tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake)) ** 2)
        da_loss = (da_loss_real1 + da_loss_fake1) / 2.0


        d_loss=da_loss+db_loss

    with tf.name_scope("A2B_discriminator_train"):
        # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_tvars_B1 = [var for var in tf.trainable_variables() if 'discriminatorB1' in var.name]
        discrim_optim_B1 = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars_B1 = discrim_optim_B1.compute_gradients(db_loss1, var_list=discrim_tvars_B1)
        discrim_train_B1 = discrim_optim_B1.apply_gradients(discrim_grads_and_vars_B1)
    with tf.name_scope("A2B_discriminator_train"):
        discrim_tvars_B2 = [var for var in tf.trainable_variables() if 'discriminatorB2' in var.name]
        discrim_optim_B2 = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars_B2 = discrim_optim_B2.compute_gradients(db_loss2, var_list=discrim_tvars_B2)
        discrim_train_B2 = discrim_optim_B2.apply_gradients(discrim_grads_and_vars_B2)


    with tf.name_scope("B2A_discriminator_train"):
        # discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_tvars_A = [var for var in tf.trainable_variables() if 'discriminatorA1' in var.name]
        discrim_optim_A = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars_A = discrim_optim_A.compute_gradients(da_loss, var_list=discrim_tvars_A)
        discrim_train_A = discrim_optim_A.apply_gradients(discrim_grads_and_vars_A)

    with tf.name_scope("A2B_generator_train"):
        with tf.control_dependencies([discrim_train_B1,discrim_train_B2]):
            gen_tvars_B = [var for var in tf.trainable_variables() if 'generator_resnet_A2B' in var.name]
            gen_optim_B = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars_B = gen_optim_B.compute_gradients(g_loss_a2b, var_list=gen_tvars_B)
            gen_train_B = gen_optim_B.apply_gradients(gen_grads_and_vars_B)



    with tf.name_scope("B2A_generator_train"):
        with tf.control_dependencies([discrim_train_A]):
            gen_tvars_A = [var for var in tf.trainable_variables() if 'generator_resnet_B2A' in var.name]
            gen_optim_A = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars_A = gen_optim_A.compute_gradients(g_loss_b2a, var_list=gen_tvars_A)
            gen_train_A = gen_optim_A.apply_gradients(gen_grads_and_vars_A)




    with tf.name_scope("A2B_generator_water_train"):

        gen_tvars_water = [var for var in tf.trainable_variables() if 'generator_water_A2B' in var.name]
        gen_optim_water = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars_water = gen_optim_water.compute_gradients(g_loss_water, var_list=gen_tvars_water)
        gen_train_water = gen_optim_water.apply_gradients(gen_grads_and_vars_water)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply(
        [g_loss_a2b, g_loss_b2a, g_loss, da_loss, db_loss1,db_loss2, g_loss_water])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        g_loss_a2b=ema.average(g_loss_a2b),
        g_loss_b2a=ema.average(g_loss_b2a),
        g_loss=ema.average(g_loss),
        da_loss=ema.average(da_loss),

        db_loss1=ema.average(db_loss1),
        db_loss2=ema.average(db_loss2),
        db_loss_real2=ema.average(db_loss_real2),
        db_loss_fake2=ema.average(db_loss_fake2),
        g_loss_water=ema.average(g_loss_water),
        discrim_grads_and_vars=discrim_grads_and_vars_B1,
        gen_grads_and_vars=gen_grads_and_vars_B,
        outputs_a=fake_a,
        output_b=fake_b,
        train=tf.group(update_losses, incr_global_step, gen_train_B,gen_train_A,gen_train_water),
        label=label1,
        fc8r=fc8rs,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def append_index_test_B(filesets,step=False):

    index_path = os.path.join(a.output_dir, "indexB.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["a_image", "outputs_b", "b_image"]:
            index.write("<td><img src='B/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def append_index_test_A(filesets,step=False):


    index_path = os.path.join(a.output_dir, "indexA.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["b_image", "outputs_a", "a_image"]:
            index.write("<td><img src='A/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def save_generated_A_test(fetches,step=None):
    image_dir = os.path.join(a.output_dir,"A")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)  # \D0½\A8\C9\FA\B3\C9\D5\D5Ƭ·\BE\B6

    filesets = []
    for i, in_path in enumerate(fetches["paths_B"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["b_image","outputs_a","a_image"]:
            # filename = name + ".png"
            filename = name  + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def save_generated_B_test(fetches,step=None):
    image_dir = os.path.join(a.output_dir, "B")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)  # \D0½\A8\C9\FA\B3\C9\D5\D5Ƭ·\BE\B6

    filesets = []
    index_list=[]
    for i , in_path in enumerate(fetches["paths_B"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        name_list=name.split('.')
        name0=name_list[0]
        index=name0[-1]
        index_list.append(index)
    for i, in_path in enumerate(fetches["paths_A"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["a_image","outputs_b","b_image"]:
            filename = name +index_list[i]+ "-" + kind + ".png"
            # filename = name +index_list[i]+ ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def save_generated_A(fetches, epoch, step=None):
    image_dir = os.path.join(a.output_dir, "generated_A" + str(epoch))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)  # \D0½\A8\C9\FA\B3\C9\D5\D5Ƭ·\BE\B6

    filesets = []
    for i, in_path in enumerate(fetches["paths_B"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["outputs_a"]:
            filename = name + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def save_generated_B(fetches, epoch, step=None):
    image_dir = os.path.join(a.output_dir, "generated_B" + str(epoch))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)  # \D0½\A8\C9\FA\B3\C9\D5\D5Ƭ·\BE\B6

    filesets = []
    index_list=[]
    for i , in_path in enumerate(fetches["paths_B"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        name_list=name.split('.')
        name0=name_list[0]
        index=name0[-1]
        index_list.append(index)
    for i, in_path in enumerate(fetches["paths_A"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["outputs_b"]:
            filename = name +index_list[i]+ ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))


    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.a_image, examples.b_image,examples.label_B,examples.d_images)
    labelp = model.label
    fc8rp = model.fc8r
    # undo colorization splitting on images that we use for display/output

    a_image = deprocess(examples.a_image)
    b_image = deprocess(examples.b_image)
    outputs_a = deprocess(model.outputs_a)
    output_b = deprocess(model.output_b)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_a"):
        converted_a_image = convert(a_image)

    with tf.name_scope("convert_b"):
        converted_b_image = convert(b_image)

    with tf.name_scope("convert_outputs_a"):
        converted_outputs_a = convert(outputs_a)

    with tf.name_scope("convert_outputs_b"):
        converted_output_b = convert(output_b)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths_A": examples.paths_A,
            "paths_B": examples.paths_B,
            "a_image": tf.map_fn(tf.image.encode_png, converted_a_image, dtype=tf.string, name="a_image_pngs"),
            "b_image": tf.map_fn(tf.image.encode_png, converted_b_image, dtype=tf.string, name="b_image_pngs"),
            "outputs_a": tf.map_fn(tf.image.encode_png, converted_outputs_a, dtype=tf.string, name="output_a_pngs"),
            "outputs_b": tf.map_fn(tf.image.encode_png, converted_output_b, dtype=tf.string, name="output_b_pngs"),
        }

    # summaries
    with tf.name_scope("a_image_summary"):
        tf.summary.image("a_image", converted_a_image)

    with tf.name_scope("b_image_summary"):
        tf.summary.image("b_image", converted_b_image)

    with tf.name_scope("outputs_a_summary"):
        tf.summary.image("outputs_a", converted_outputs_a)

    with tf.name_scope("outputs_b_summary"):
        tf.summary.image("outputs_b", converted_output_b)



    #tf.summary.scalar("d_loss", model.d_loss)
    tf.summary.scalar("g_loss", model.g_loss)
    tf.summary.scalar("g_loss_a2b", model.g_loss_a2b)
    tf.summary.scalar("g_loss_b2a", model.g_loss_b2a)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    # for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
    #     tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=0)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                # filesets = save_images(results)
                filesets_A = save_generated_A_test(results)
                filesets_B= save_generated_B_test(results)
                # for i, f in enumerate(filesets):
                    # print("evaluated image", f["name"])
                index_path_B = append_index_test_B(filesets_B)
                index_path_A = append_index_test_A(filesets_A)

            print("wrote index at", index_path_B)
            print("wrote index at", index_path_A)
        else:
            # training
            start = time.time()

            max_steps = examples.count * 100
            cnt = 0
            for step in range(max_steps):
                mod = cnt % (examples.count )
                if mod < examples.count and cnt >= examples.count :
                    results_generated = sess.run(display_fetches)
                    _ = save_generated_A(results_generated, int(cnt / (examples.count )))
                    _ = save_generated_B(results_generated, int(cnt / (examples.count )))
                cnt += 1

                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                #print("lsble_count =", sess.run(labelp))
                #print("frc_count =", sess.run(fc8rp))

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    #fetches["d_loss"] = model.d_loss
                    fetches["db_loss1"] = model.db_loss1
                    fetches["db_loss2"] = model.db_loss2
                    fetches["da_loss"] = model.da_loss
                    fetches["g_loss"] = model.g_loss
                    fetches["g_loss_a2b"] = model.g_loss_a2b
                    fetches["g_loss_b2a"] = model.g_loss_b2a
                    fetches["db_loss_real2"] = model.db_loss_real2
                    fetches["db_loss_fake2"] = model.db_loss_fake2

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    #print("d_loss", results["d_loss"])
                    print("da_loss", results["da_loss"])
                    print("db_loss1", results["db_loss1"])
                    print("db_loss2", results["db_loss2"])
                    print("g_loss", results["g_loss"])
                    print("g_loss_a2b", results["g_loss_a2b"])
                    print("g_loss_b2a", results["g_loss_b2a"])
                    print("db_loss_fake2", results["g_loss_b2a"])
                    print("db_loss_real2", results["g_loss_b2a"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()