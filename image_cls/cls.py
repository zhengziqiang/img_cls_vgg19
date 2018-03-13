from __future__ import division
import glob
from pylab import *
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from PIL import Image
import tensorflow.contrib.slim as slim
import numpy as np
import argparse
from functools import reduce
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir",required=True, help="path to folder containing train images")
parser.add_argument("--val_dir",required=True, help="path to folder containing val images")
parser.add_argument("--test_internal",type=int,default=50, help="number of test internal")
parser.add_argument("--max_epochs", type=int,default=200, help="number of training epochs")
parser.add_argument("--image_size", type=int,default=256, help="image size")
parser.add_argument("--lr", type=float,default=0.0002, help="learning rate")
a = parser.parse_args()
VGG_MEAN = [103.939, 116.779, 123.68]
class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [256, 256, 1]
        assert green.get_shape().as_list()[1:] == [256, 256, 1]
        assert blue.get_shape().as_list()[1:] == [256, 256, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [256, 256, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 32768, 4096, "fc6")  # 25088 = ((256 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 4, "fc8")

        return self.fc8
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
sess=tf.Session()


class VGG16:
    def __init__(self):
        pass

    def build(self, input, is_training=True):
        """
        input is the placeholder of tensorflow
        build() assembles vgg16 network
        """

        # flag: is_training? for tensorflow-graph
        self.train_phase = tf.constant(is_training) if is_training else None

        self.conv1_1 = self.convolution(input, 'conv1_1')
        self.conv1_2 = self.convolution(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling(self.conv1_2, 'pool1')

        self.conv2_1 = self.convolution(self.pool1, 'conv2_1')
        self.conv2_2 = self.convolution(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling(self.conv2_2, 'pool2')

        self.conv3_1 = self.convolution(self.pool2, 'conv3_1')
        self.conv3_2 = self.convolution(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.convolution(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling(self.conv3_3, 'pool3')

        self.conv4_1 = self.convolution(self.pool3, 'conv4_1')
        self.conv4_2 = self.convolution(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.convolution(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling(self.conv4_3, 'pool4')

        self.conv5_1 = self.convolution(self.pool4, 'conv5_1')
        self.conv5_2 = self.convolution(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.convolution(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling(self.conv5_3, 'pool5')

        self.fc6 = self.fully_connection(self.pool5, Activation.relu, 'cifar')
        # self.fc7 = self.fully_connection(self.fc6, Activation.relu, 'fc7')
        # self.fc8 = self.fully_connection(self.fc7, Activation.softmax, 'fc8')

        self.prob = self.fc6

        return self.prob

    def pooling(self, input, name):
        """
        Args: output of just before layer
        Return: max_pooling layer
        """
        return tf.nn.max_pool(input, ksize=vgg.ksize, strides=vgg.pool_strides, padding='SAME', name=name)

    def convolution(self, input, name):
        """
        Args: output of just before layer
        Return: convolution layer
        """
        print('Current input size in convolution layer is: ' + str(input.get_shape().as_list()))
        with tf.variable_scope(name):
            size = vgg.structure[name]
            kernel = self.get_weight(size[0], name='w_' + name)
            bias = self.get_bias(size[1], name='b_' + name)
            conv = tf.nn.conv2d(input, kernel, strides=vgg.conv_strides, padding='SAME', name=name)
            out = tf.nn.relu(tf.add(conv, bias))
        return self.batch_normalization(out)

    def fully_connection(self, input, activation, name):
        """
        Args: output of just before layer
        Return: fully_connected layer
        """
        size = vgg.structure[name]
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = reduce(lambda x, y: x * y, shape[1:])
            x = tf.reshape(input, [-1, dim])

            weights = self.get_weight([dim, size[0][0]], name=name)
            biases = self.get_bias(size[1], name=name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            fc = activation(fc)

            print('Input shape is: ' + str(shape))
            print('Total nuron count is: ' + str(dim))

            return self.batch_normalization(fc)

    def batch_normalization(self, input, decay=0.9, eps=1e-5):
        """
        Batch Normalization
        Result in:
            * Reduce DropOut
            * Sparse Dependencies on Initial-value(e.g. weight, bias)
            * Accelerate Convergence
            * Enable to increase training rate
        Args: output of convolution or fully-connection layer
        Returns: Normalized batch
        """
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        beta = tf.Variable(tf.zeros([n_out]))
        gamma = tf.Variable(tf.ones([n_out]))

        if len(shape) == 2:
            batch_mean, batch_var = tf.nn.moments(input, [0])
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)

    def get_weight(self, shape, name):
        """
        generate weight tensor
        Args: weight size
        Return: initialized weight tensor
        """
        initial = tf.truncated_normal(shape, 0.0, 1.0) * 0.01
        return tf.Variable(initial, name='w_' + name)

    def get_bias(self, shape, name):
        """
        generate bias tensor
        Args: bias size
        Return: initialized bias tensor
        """


        return tf.Variable(tf.truncated_normal(shape, 0.0, 1.0) * 0.01, name='b_' + name)

with tf.variable_scope(tf.get_variable_scope()):
    real_image=tf.placeholder(tf.float32,[1,256,256,3])
    label_zero=tf.placeholder(tf.int32,[None])
    label=tf.one_hot(label_zero,4)
    vgg_net=Vgg19()
    logits=vgg_net.build(real_image)
    index=tf.argmax(logits,1)
    loss=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label)

optimizer = tf.train.GradientDescentOptimizer(a.lr).minimize(loss)
sess.run(tf.global_variables_initializer())

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    for epoch in range(a.max_epochs):
        loss_epoch=0.0
        file_cnt=0
        error_average=0.0
        if len(glob.glob(a.train_dir+"/*.jpg"))==0:
            raise Exception("train_dir contains no image files")

        for files in glob.glob(a.train_dir+"/*.jpg"):
            img=Image.open(files)
            img_resize=img.resize((a.image_size,a.image_size))
            img_r_array=array(img_resize)
            img_r=np.reshape(img_r_array,[1,256,256,3])
            p,n=os.path.split(files)
            nn=n.split('.')
            name=nn[0]
            number=np.int32(name[-1])
            number=np.reshape(number,[1])
            # label_onehot=tf.one_hot(number,4)
            _,cost_file=sess.run([optimizer,loss],feed_dict={real_image:img_r,label_zero:number})
            error_average+=cost_file
            if file_cnt%a.test_internal==0:
                print ("epoch:%d    error_average_50:%.4f"% (epoch,error_average/50.0))
                error_average=0.0
            file_cnt+=1
            loss_epoch+=cost_file
        cnt=0
        if epoch%1==0 and epoch >0:
            print("epoch_loss:%.4f" % loss_epoch)
            if len(glob.glob(a.val_dir + "/*.jpg")) == 0:
                raise Exception("val_dir contains no image files")
            for files in glob.glob(a.val_dir+"/*.jpg"):
                img = Image.open(files)
                img_resize = img.resize((a.image_size, a.image_size))
                img_r_array = array(img_resize)
                img_r = np.reshape(img_r_array, [1, 256, 256, 3])
                my_list = sess.run([index],feed_dict={real_image: img_r})
                my_list=np.reshape(my_list,[1])
                p, n = os.path.split(files)
                nn = n.split('.')
                name = nn[0]
                number = np.int(name[-1])
                if number==my_list[0]:
                    cnt += 1
            acc=cnt/np.float(len(glob.glob(a.val_dir+"/*.jpg")))
            print("accuracy:%.4f"%acc)
