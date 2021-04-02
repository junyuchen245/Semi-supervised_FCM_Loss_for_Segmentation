import tensorflow as tf
from tensorflow.keras import backend as K

class RFCM_loss():
    def __init__(self, fuzzy_factor=2, regularizer_wt=0.):
        '''
        Unsupervised Robust Fuzzy C-mean loss function for ConvNet based image segmentation

        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).

        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2
        :param regularizer_wt: weighting parameter for regularization, default value = 0

        Note that ground truth segmentation is NOT needed in this loss fuction, instead, the input image is required.
        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param image: input image to the ConvNet.
        '''
        self.fuzzy_factor = fuzzy_factor
        self.wt = regularizer_wt

    def rfcm_loss_func(self, image, y_pred):
        dim = len(image.get_shape().as_list()[1:-1])
        assert dim == 3 or dim == 2, 'Supports only 3D or 2D!'
        num_clus = y_pred.get_shape().as_list()[-1]
        if dim == 3:
            k = tf.constant([[
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],[
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ],[
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]], dtype=tf.float32, name='k')
            kernel = tf.reshape(k, [3, 3, 3, 1, 1], name='kernel')
        else:
            k = tf.constant([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ], dtype=tf.float32, name='k')
            kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')

        img = K.reshape(image, (-1, K.prod(K.shape(image)[1:])))
        seg = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:-1]), num_clus))
        J_1 = 0
        J_2 = 0
        for i in range(num_clus):
            mem = K.pow(seg[..., i], self.fuzzy_factor)
            v_k = tf.reduce_sum(tf.math.multiply(img, mem))/tf.reduce_sum(mem)
            J_1 += tf.math.multiply(mem, K.square(img - v_k))
            J_in = 0
            for j in range(num_clus):
                if i==j:
                    continue
                mem_j = K.pow(seg[..., j], self.fuzzy_factor)
                if dim == 2:
                    mem_j = K.reshape(mem_j, (-1, K.shape(y_pred)[1], K.shape(y_pred)[2], 1))
                    res = tf.nn.conv2d(mem_j, kernel, [1, 1, 1, 1], "SAME")
                else:
                    mem_j = K.reshape(mem_j, (-1, K.shape(y_pred)[1], K.shape(y_pred)[2], K.shape(y_pred)[3], 1))
                    res = tf.nn.conv2d(mem_j, kernel, [1, 1, 1, 1, 1], "SAME")
                res = K.reshape(res, (-1, K.prod(K.shape(image)[1:])))
                J_in += res
            J_2 += tf.math.multiply(mem, J_in)
        return tf.reduce_mean(J_1)+self.wt*tf.reduce_mean(J_2)

    def loss(self, I, J):
        return self.rfcm_loss_func(I, J)

class FCM_label():
    def __init__(self, fuzzy_factor=2):
        '''
        Supervised Fuzzy C-mean loss function for ConvNet based image segmentation

        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).

        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2

        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param y_true: ground thruth label
        '''
        self.fuzzy_factor = fuzzy_factor

    def fcm_loss_func(self, y_true, y_pred):
        num_clus = y_pred.get_shape().as_list()[-1]
        labels = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:-1]), num_clus))
        seg = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:-1]), num_clus))
        J_1 = 0
        for i in range(num_clus):
            label = labels[..., i]
            mem = K.pow(seg[..., i], self.fuzzy_factor)
            J_1 += tf.math.multiply(mem, K.square(label - 1))
        return tf.reduce_mean(J_1)

    def loss(self, I, J):
        return self.fcm_loss_func(I, J)