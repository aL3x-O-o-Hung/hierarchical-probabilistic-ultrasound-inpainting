from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.ops import math_ops

r=8
print(tf.__version__)




class BatchNormRelu(tf.keras.layers.Layer):
    """Batch normalization + ReLu"""

    def __init__(self, name=None, dtype=None):
        super(BatchNormRelu, self).__init__(name=name)
        self.bnorm = tf.keras.layers.experimental.SyncBatchNormalization(momentum=0.9, dtype=dtype)

    def call(self, inputs, is_training=True):
        x = self.bnorm(inputs, training=is_training)
        x = tf.keras.activations.swish(x)
        return x




class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Conv2D Fixed Padding layer"""

    def __init__(self, filters, kernel_size, stride,dilation, name=None, dtype=None):
        super(Conv2DFixedPadding, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=1,
            dilation_rate=dilation,
            padding=('same' if stride == 1 else 'valid'),
            activation=None,
            dtype=dtype
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        return x

class Conv2DTranspose(tf.keras.layers.Layer):
    """Conv2DTranspose layer"""

    def __init__(self, output_channels, kernel_size, name=None, dtype=None):
        super(Conv2DTranspose, self).__init__(name=name)

        self.tconv1 = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            activation=None,
            dtype=dtype
        )
        self.conv=Conv2DFixedPadding(filters=output_channels,dilation=1,kernel_size=kernel_size,stride=1)

    def call(self, inputs,is_training=True):
        x = self.tconv1(inputs)
        x=self.conv(x)
        return x

class SEBlock(tf.keras.layers.Layer):
    """squeeze and excitation"""
    def __init__(self,channel_original,channel_new,name=None):
        super(SEBlock,self).__init__(name=name)
        self.pool=tf.keras.layers.GlobalAveragePooling2D()
        self.conv1=Conv2DFixedPadding(filters=channel_new,kernel_size=1,stride=1,dilation=1)
        self.conv2=Conv2DFixedPadding(filters=channel_original,kernel_size=1,stride=1,dilation=1)
    def call(self,inputs):
        x=self.pool(inputs)
        x=tf.expand_dims(x,axis=1)
        x=tf.expand_dims(x,axis=1)
        x=self.conv1(x)
        x=tf.keras.activations.relu(x)
        x=self.conv2(x)
        x=tf.keras.activations.sigmoid(x)
        x=x*inputs
        return x

class ChannelAttention(tf.keras.layers.Layer):
    """Cahnnelwise attention for CBAM"""
    def __init__(self,channel_original,channel_new,name=None):
        super(ChannelAttention,self).__init__(name=name)
        self.avgpool=tf.keras.layers.GlobalAveragePooling2D()
        self.maxpool=tf.keras.layers.GlobalMaxPooling2D()
        self.dense1=tf.keras.layers.Dense(channel_new,activation=tf.keras.activations.relu)
        self.dense2=tf.keras.layers.Dense(channel_original,activation=tf.keras.activations.relu)

    def call(self,inputs):
        x=inputs
        avg=self.avgpool(x)
        max=self.maxpool(x)
        avg=self.dense1(avg)
        avg=self.dense2(avg)
        max=self.dense1(max)
        max=self.dense2(max)
        x=avg+max
        x=tf.keras.activations.sigmoid(x)
        x=tf.expand_dims(x,axis=1)
        x=tf.expand_dims(x,axis=1)
        return x


class SpatialAttention(tf.keras.layers.Layer):
    """spatial attention for CBAM"""
    def __init__(self,name=None):
        super(SpatialAttention,self).__init__(name=name)
        self.conv=Conv2DFixedPadding(filters=1,
                                        kernel_size=3,
                                        dilation=1,
                                        stride=1)

    def call(self,inputs):
        x=inputs
        s=x.get_shape().as_list()[3]
        avg=tf.reduce_sum(x,axis=-1,keepdims=True)/s
        max=tf.reduce_max(x,axis=-1,keepdims=True)
        x=tf.concat([avg,max],axis=-1)
        x=self.conv(x)
        x=tf.keras.activations.sigmoid(x)
        return x

class CBAMBlock(tf.keras.layers.Layer):
    """CBAM attention"""
    def __init__(self,channel_original,channel_new,name=None):
        super(CBAMBlock,self).__init__(name=name)
        self.channel=ChannelAttention(channel_original,channel_new,name=name+'_channel_attention')
        self.spatial=SpatialAttention(name=name+'_spatial_attention')
    def call(self,inputs):
        x=inputs
        x=x*self.channel(x)
        x=x*self.spatial(x)
        return x

class ConvBlock(tf.keras.layers.Layer):
    """Downsampling ConvBlock on Encoder side"""

    def __init__(self, filters, dilation,kernel_size=3, do_max_pool=True, additional_block=None, name=None):
        super(ConvBlock, self).__init__(name=name)
        self.do_max_pool = do_max_pool
        self.conv1 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        stride=1)
        self.brelu1 = BatchNormRelu()
        self.conv2 = Conv2DFixedPadding(filters=filters,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        stride=1)
        self.brelu2 = BatchNormRelu()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2,
                                                  strides=2,
                                                  padding='valid')
        self.additional_block=additional_block
        if additional_block=='SE':
            self.ad1=SEBlock(filters,filters//r,name=name+'SE1')
            self.ad2=SEBlock(filters,filters//r,name=name+'SE2')
        elif additional_block=='CBAM':
            self.ad1=CBAMBlock(filters,filters//r,name=name+'CBAM1')
            self.ad2=CBAMBlock(filters,filters//r,name=name+'CBAM2')


    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        x = self.brelu1(x, is_training)
        if self.additional_block is not None:
            x=self.ad1(x)
        x = self.conv2(x)
        x = self.brelu2(x, is_training)
        if self.additional_block is not None:
            x=self.ad2(x)
        output_b = x
        if self.do_max_pool:
            x = self.max_pool(x)
        return x, output_b


class DeConvBlock(tf.keras.layers.Layer):
    """Upsampling DeConvBlock on Decoder side"""

    def __init__(self, filters,dilation, kernel_size=2,additional_block=None, name=None):
        super(DeConvBlock, self).__init__(name=name)
        self.dilation=dilation
        if dilation==1:
            self.tconv1 = Conv2DTranspose(output_channels=filters,
                                          kernel_size=kernel_size)
            self.conv1 = Conv2DFixedPadding(filters=filters,
                                            kernel_size=3,
                                            dilation=1,
                                            stride=1,
                                            name=name+'_conv1')
            self.brelu1 = BatchNormRelu()
        self.conv2=ConvBlock(filters=filters,dilation=dilation,additional_block=additional_block,do_max_pool=False, name=name + '_conv2')

    def call(self, inputs, output_b, is_training=True):
        x=inputs
        if self.dilation==1:
            x = self.tconv1(x,is_training=is_training)
            x = tf.keras.activations.swish(x)
            x=self.conv1(x)
            x=self.brelu1(x,is_training=is_training)
        x = tf.concat([output_b, x], axis=-1)

        x,_ = self.conv2(x,is_training=is_training)
        return x


class PriorBlock(tf.keras.layers.Layer):
    """calculating Prior Block"""

    def __init__(self, filters, name=None):  # filters: number of the layers incorporated into the decoder
        super(PriorBlock, self).__init__(name=name)
        self.conv = Conv2DFixedPadding(filters=filters * 2, kernel_size=1, stride=1,dilation=1)

    def call(self, inputs):
        x = 0.1 * self.conv(inputs)
        s = x.get_shape().as_list()[3]
        mean = x[:, :, :, :s // 2]
        mean =tf.keras.activations.tanh(mean)
        logstd = x[:, :, :, s // 2:]
        logstd = 3.0 * tf.keras.activations.tanh(logstd)
        std = K.exp(logstd)
        # var = K.abs(logvar)
        return tf.concat([mean, std], axis=-1)


@tf.function
def prob_function(inputs):
    ts = inputs.get_shape()
    s = ts.as_list()
    s[3] = int(s[3] / 2)
    dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    if ts[0] is None:
        samp = dist.sample([1, s[1], s[2], s[3]])
    else:
        samp = dist.sample([ts[0], s[1], s[2], s[3]])
    dis = 0.5 * tf.math.multiply(samp, inputs[:, :, :, s[3]:])
    dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
    return dis


class Prob(tf.keras.layers.Layer):
    """sample from the gaussian distribution"""
    def __init__(self, name=None):
        super(Prob, self).__init__(name=name)

    def call(self, inputs):
        ts = inputs.get_shape()
        s = ts.as_list()
        s[3] = int(s[3] / 2)
        dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
        if ts[0] is None:
            samp = dist.sample([1, s[1], s[2], s[3]])
        else:
            samp = dist.sample([ts[0], s[1], s[2], s[3]])
        dis = tf.math.multiply(samp, inputs[:, :, :, s[3]:])
        dis = tf.math.add(dis, inputs[:, :, :, 0:s[3]])
        return dis


class Encoder(tf.keras.layers.Layer):
    """encoder of the network"""

    def __init__(self, num_layers, num_filters,dilation,additional_block=None, name=None):
        super(Encoder, self).__init__(name=name)
        self.convs = []
        for i in range(num_layers):
            if i < num_layers - 1:
                max_pool=True
                if dilation[i]>1:
                    max_pool=False
                conv_temp = ConvBlock(filters=num_filters[i],dilation=dilation[i],additional_block=additional_block,do_max_pool=max_pool, name=name + '_conv' + str(i + 1))
            else:
                conv_temp = ConvBlock(filters=num_filters[i], dilation=dilation[i], additional_block=additional_block,do_max_pool=False, name=name + '_conv' + str(i + 1))
            self.convs.append(conv_temp)

    def call(self, inputs, is_training=True):
        list_b = []
        x = inputs
        for i in range(len(self.convs)):
            x, b = self.convs[i](x, is_training=is_training)
            list_b.append(b)
        return x, list_b


class DecoderWithPriorBlockPosterior(tf.keras.layers.Layer):
    """decoder of the network with prior block in Posterior"""

    def __init__(self, num_layers, num_filters, num_filters_prior,dilation,additional_block=None, name=None):
        super(DecoderWithPriorBlockPosterior, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i],dilation[i],additional_block=additional_block, name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + 'prior' + str(i)))

    def call(self, inputs, blocks, is_training=True):
        x = inputs
        prior = []
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prior.append(p)
            if i != self.num_layers - 1:
                x = tf.concat([x, p], axis=-1)
                x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return prior


class DecoderWithPriorBlock(tf.keras.layers.Layer):
    """decoder of the network with prior block"""

    def __init__(self, num_layers, num_filters, num_filters_prior, dilation,additional_block=None,name=None):
        super(DecoderWithPriorBlock, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_filters_prior = num_filters_prior
        self.deconvs = []
        self.priors = []
        self.prob_function = Prob()
        for i in range(num_layers):
            self.deconvs.append(DeConvBlock(num_filters[i],dilation[i],additional_block=additional_block, name=name + '_dconv' + str(i)))
            self.priors.append(PriorBlock(num_filters_prior[i], name=name + '_prior' + str(i)))

    def call(self, inputs, blocks, posterior_delta, is_training=True):
        x = inputs
        prior = []
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prior.append(p)
            s = p.get_shape().as_list()[3]
            posterior = tf.concat([
                prior[i][:, :, :, :s // 2] + posterior_delta[i][:, :, :, :s // 2],
                prior[i][:, :, :, s // 2:] * posterior_delta[i][:, :, :, s // 2:],
            ], axis=-1)
            prob = self.prob_function(posterior)
            x = tf.concat([x, prob], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x, prior

    def sample(self, x, blocks, is_training=False):
        for i in range(self.num_layers):
            p = self.priors[i](x)
            prob = prob_function(p)
            x = tf.concat([x, prob], axis=-1)
            x = self.deconvs[i](x, blocks[i], is_training=is_training)
        return x

class Decoder(tf.keras.layers.Layer):
    """decoder of the network"""

    def __init__(self, num_layers, num_prior_layers, num_filters, num_filters_in_prior, num_filters_prior,dilation,dilation_in_prior,additional_block=None, name=None):
        """
        :param num_layers: number of layers in the non-prior part
        :param num_prior_layers: number of layers in the prior part
        :param num_filters: list of numbers of filters in different layers in non-prior part
        :param num_filters_in_prior: list of numbers of filters in different layers in non-prior part
        :param num_filters_prior: list of numbers of priors in different prior blocks
        :param dilation: list of dilation rates in the non-prior part
        :param dilation_in_prior: list of dilation rates in prior part
        :param additional_block: using SE or CBAM or None
        :param name: name
        """
        super(Decoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters_prior = num_filters_prior
        self.num_filters = num_filters
        self.num_filters_in_prior = num_filters_in_prior
        self.num_prior_layers = num_prior_layers
        self.prior_decode = DecoderWithPriorBlock(num_prior_layers,
                                                  num_filters_in_prior,
                                                  num_filters_prior,
                                                  dilation=dilation_in_prior,
                                                  additional_block=additional_block,
                                                  name=name + '_with_prior')
        self.tconvs = []
        self.generate = []
        for i in range(num_layers):
            self.tconvs.append(DeConvBlock(num_filters[i],dilation=dilation[i],additional_block=additional_block, name=name + '_without_prior'))

    def call(self, inputs, b, posterior_delta, is_training=True):
        x = inputs
        x, prior = self.prior_decode(x, b[0:self.num_prior_layers], posterior_delta, is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x, prior

    def sample(self, x, b, is_training=False):
        x = self.prior_decode.sample(x, b[0:self.num_prior_layers], is_training=is_training)
        for i in range(self.num_layers):
            x = self.tconvs[i](x, b[self.num_prior_layers + i], is_training=is_training)
        return x


####KL divergence####
def residual_kl_gauss(y_delta, y_prior):
    # See NVAE paper
    s = y_delta.get_shape().as_list()[3]
    mean_delta = y_delta[:, :, :, 0:s // 2]
    std_delta = y_delta[:, :, :, s // 2:]
    # mean_prior = y_prior[:, :, :, 0:s // 2]
    std_prior = y_prior[:, :, :, s // 2:]
    first = math_ops.log(std_delta)
    second = 0.5 * math_ops.divide(K.square(mean_delta), K.square(std_prior))
    third = 0.5 * K.square(std_delta)
    loss = second + third - first - 0.5
    loss = tf.reduce_mean(loss)
    return loss

def residual_kl_normal(y_prior):
    s=y_prior.get_shape().as_list()[3]
    mean_prior=y_prior[:,:,:,0:s//2]
    std_prior=y_prior[:,:,:,s//2:]
    mean_delta=mean_prior
    std_delta=std_prior
    first=math_ops.log(std_delta+0.0001)
    second=0.5*math_ops.divide(K.square(mean_delta+0.0001),K.square(std_prior+0.0001))
    third=0.5*K.square(std_delta)
    loss=second+third-first-0.5
    loss=tf.reduce_mean(loss)
    return loss



####final model####
class HierarchicalProbUNet(tf.keras.Model):
    def __init__(self, num_layers, num_filters, num_prior_layers, num_filters_prior,dilation, rec, p, s, tv,additional_block=None, name=None):
        '''
        :param num_layers: an integer, number of layers in the encoder side
        :param num_filters: a list, number of filters at each layer
        :param num_prior_layers: an integer, number of layers with prior blocks
        :param num_filters_prior: a list, number of filters at each prior blocks
        :param dilation: a list, dilation rates of each layer in the encoder side
        :param rec: a float, weight of reconstruction loss
        :param p: a list, weights of perceptual loss at each VGG layer
        :param s: a list, weights of style loss at each VGG layer
        :param tv: a float, weight of total variation loss (will be removed in the future)
        :param additional_block: using SE or CBAM or None
        :param name: name
        '''
        super(HierarchicalProbUNet, self).__init__(name=name)
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.num_prior_layers = num_prior_layers
        self.num_filters_decoder = num_filters[::-1][1:]
        self.num_filters_prior = num_filters_prior
        self.additional_block=additional_block
        self.dilation=dilation
        self.dilation_decoder=dilation[::-1][1:]
        self.encoder = Encoder(num_layers,
                               num_filters,
                               dilation=dilation,
                               additional_block=additional_block,
                               name=name + '_encoder')
        self.encoder_post = Encoder(num_layers,
                                    num_filters,
                                    dilation=dilation,
                                    additional_block=additional_block,
                                    name=name + '_encoder_post')
        self.decoder = Decoder(num_layers=num_layers - num_prior_layers - 1,
                               num_prior_layers=num_prior_layers,
                               num_filters=self.num_filters_decoder[num_prior_layers:],
                               num_filters_in_prior=self.num_filters_decoder[:num_prior_layers],
                               num_filters_prior=num_filters_prior,
                               dilation=self.dilation_decoder[num_prior_layers:],
                               dilation_in_prior=self.dilation_decoder[:num_prior_layers],
                               additional_block=additional_block,
                               name=name + '_decoder')
        self.decoder_post = DecoderWithPriorBlockPosterior(num_prior_layers,
                                                           self.num_filters_decoder[:num_prior_layers],
                                                           num_filters_prior,
                                                           dilation=self.dilation_decoder[:num_prior_layers],
                                                           additional_block=additional_block,
                                                           name=name + '_decoder_post')
        self.conv = Conv2DFixedPadding(filters=1, kernel_size=1, stride=1, dilation=1,name=name + '_conv_final')
        self.VGG = VGG16()
        for layer in self.VGG.layers:
            layer.trainable = False
        self.VGGs = []
        for i in range(1, 6):
            if p[i - 1] > 0 or s[i - 1] > 0:
                name = 'block' + str(i) + '_conv1'
                vgg_input = self.VGG.input
                vgg_out = self.VGG.get_layer(name).output
                self.VGGs.append(tf.keras.Model(inputs=vgg_input, outputs=vgg_out))
                for layer in self.VGGs[i - 1].layers:
                    layer.trainable = False
            else:
                self.VGGs.append(None)
        self.rec = rec
        self.p = p
        self.s = s
        self.tv = tv
        self.upsamp = tf.keras.layers.UpSampling2D()
        self.downsamp = tf.keras.layers.AveragePooling2D()

    def get_gaussian_pyramid(self,mask,l):
        masks=[mask]
        for i in range(l-1):
            mask=tfa.image.gaussian_filter2d(mask)
            mask=self.downsamp(mask)
            masks.append(mask)
        masks=masks[::-1]
        return masks

    def get_laplacian_pyramid(self,p):
        py=[]
        for i in range(len(p)):
            if i==0:
                py.append(p[i])
            else:
                py.append(p[i]-tfa.image.gaussian_filter2d(self.upsamp(p[i-1])))
        return py

    def build_from_pyramid(self,lp_input,lp_result,masks):
        final_res=[]
        for i in range(len(lp_input)):
            if i==0:
                final_res.append(lp_result[i]*masks[i]+lp_input[i]*(1-masks[i]))
            else:
                final_res.append(lp_result[i]*masks[i]+lp_input[i]*(1-masks[i])+tfa.image.gaussian_filter2d(self.upsamp(final_res[i-1])))
        return final_res[len(lp_input)-1]

    def reconstruction_loss(self,y_true,y_pred,m,weight):
        l=4
        tg=self.get_gaussian_pyramid(y_true,l)
        pg=self.get_gaussian_pyramid(y_pred,l)
        mm=self.get_gaussian_pyramid(m,l)
        tl=self.get_laplacian_pyramid(tg)
        pl=self.get_laplacian_pyramid(pg)
        for i in range(l):
            if i==0:
                loss=1/(1.5**i)*K.square(tl[i]-pl[i])*K.pow(1.1,-mm[i])
                loss=weight*tf.reduce_mean(loss)
                self.add_metric(loss,name='reconstruction loss'+str(i),aggregation='mean')
            else:
                temp_loss=1/(1.5**i)*K.square(tl[i]-pl[i])*K.pow(1.1,-mm[i])
                temp_loss=weight*tf.reduce_mean(temp_loss)
                self.add_metric(temp_loss,name='reconstruction loss'+str(i),aggregation='mean')
                loss+=temp_loss
                #loss=loss/(2**l-1)
        return loss

    def perceptual_loss(self,y_true,y_pred,l,weight):
        loss=K.square(y_true-y_pred)
        loss=weight*tf.reduce_mean(loss)
        self.add_metric(loss,name='perceptual loss'+str(l),aggregation='mean')
        return loss

    def gram_matrix(self,x):
        x=K.permute_dimensions(x,(0,3,1,2))
        shape=K.shape(x)
        B,C,H,W=shape[0],shape[1],shape[2],shape[3]
        features=K.reshape(x,K.stack([B,C,H*W]))
        gram=K.batch_dot(features,features,axes=2)
        denominator=C*H*W
        gram=gram/(K.cast(denominator,x.dtype))
        return gram

    def style_loss(self,y_true,y_pred,l,weight):
        y_true=self.gram_matrix(y_true)
        y_pred=self.gram_matrix(y_pred)
        loss=0.5*K.square(y_true-y_pred)
        loss=weight*tf.reduce_mean(loss)
        self.add_metric(loss,name='style loss'+str(l),aggregation='mean')
        return loss

    def deep_loss(self,y_true,y_pred,VGG_model,p,s):
        y_true=tf.concat((y_true,y_true,y_true),axis=-1)
        y_pred=tf.concat((y_pred,y_pred,y_pred),axis=-1)
        y_true=tf.image.resize(y_true,[224,224])
        y_true=preprocess_input(y_true*255)
        y_pred=tf.image.resize(y_pred,[224,224])
        y_pred=preprocess_input(y_pred*255)
        loss=0
        for i in range(len(VGG_model)):
            if p[i]>0 or s[i]>0:
                y_true_=VGG_model[i](y_true)
                y_pred_=VGG_model[i](y_pred)
                loss+=self.perceptual_loss(y_true_,y_pred_,i,p[i])+self.style_loss(y_true_,y_pred_,i,s[i])
        return loss

    def total_variation_loss(self,y_pred,weight):
        loss=weight*tf.reduce_mean(tf.image.total_variation(y_pred))
        self.add_metric(loss,name='tv loss',aggregation='mean')
        return loss

    def total_loss_(self,y_true,y_pred,VGG_model,rec,p,s,tv,m):
        loss=self.deep_loss(y_true,y_pred,VGG_model,p,s)
        if rec!=0:
            loss+=self.reconstruction_loss(y_true,y_pred,m,rec)
        if tv>0:
            loss+=self.total_variation_loss(y_pred,tv)
        return loss

    def training_loss(self,y_true,y_pred,VGG_model,rec,p,s,tv,m):
        loss=self.total_loss_(y_true,y_pred,VGG_model,rec,p,s,tv,m)
        return loss

    def call(self,inputs,is_training=True):
        x1=inputs[:,:,:,0:1]
        x2=inputs[:,:,:,2:3]
        m=inputs[:,:,:,3:4]
        original_input_x=x1
        ground_truth_x=x2
        mask=inputs[:,:,:,1:2]
        mask_=inputs[:,:,:,4:5]
        x1=tf.concat((x1,mask),axis=-1)
        x2=tf.concat((x2,mask),axis=-1)
        x1,b_list1=self.encoder(x1,is_training=is_training)
        b_list1=b_list1[0:-1]
        x2,b_list2=self.encoder_post(x2,is_training=is_training)
        b_list2=b_list2[0:-1]
        b_list1.reverse()
        b_list2.reverse()
        posterior_delta=self.decoder_post(x2,b_list2[0:self.num_prior_layers],is_training=is_training)
        x1,prior=self.decoder(x1,b_list1,posterior_delta,is_training=is_training)
        x1=self.conv(x1)
        x1=tf.keras.activations.sigmoid(x1)
        masks = self.get_gaussian_pyramid(mask_, 4)
        res_gaus = self.get_gaussian_pyramid(x1, 4)
        inp_gaus = self.get_gaussian_pyramid(original_input_x, 4)
        res_ = self.get_laplacian_pyramid(res_gaus)
        inp_ = self.get_laplacian_pyramid(inp_gaus)
        x1 = self.build_from_pyramid(inp_, res_, masks)
        loss=self.training_loss(ground_truth_x,x1,self.VGGs,self.rec,self.p,self.s,self.tv,m)
        ####compute kl divergence####
        for i in range(len(prior)):
            if i==0:
                s=prior[i].get_shape().as_list()[3]
                l_gauss=residual_kl_gauss(posterior_delta[i],prior[i])* (4**i)
                posterior=tf.concat([
                    prior[i][:,:,:,:s//2]+posterior_delta[i][:,:,:,:s//2],
                    prior[i][:,:,:,s//2:]*posterior_delta[i][:,:,:,s//2:],
                ],axis=-1)
                l_norm=residual_kl_normal(posterior)* (4**i)
                self.add_metric(l_gauss,name='kl_gauss'+str(i),aggregation='mean')
                self.add_metric(l_norm,name='kl_normal'+str(i),aggregation='mean')
            else:
                s=prior[i].get_shape().as_list()[3]
                temp_gauss=residual_kl_gauss(posterior_delta[i],prior[i])* (4**i)
                posterior=tf.concat([
                    prior[i][:,:,:,:s//2]+posterior_delta[i][:,:,:,:s//2],
                    prior[i][:,:,:,s//2:]*posterior_delta[i][:,:,:,s//2:],
                ],axis=-1)
                temp_norm=residual_kl_normal(posterior)* (4**i)
                self.add_metric(temp_gauss,name='kl_gauss'+str(i),aggregation='mean')
                self.add_metric(temp_norm,name='kl_normal'+str(i),aggregation='mean')
                l_gauss=math_ops.add(l_gauss,temp_gauss)
                l_norm=math_ops.add(l_norm,temp_norm)

        if self.num_prior_layers==0:
            self.add_loss(loss)
        else:
            self.add_loss(loss+l_gauss+l_norm)
        return x1

    def sample(self, inputs, is_training=False):
        x, b_list = self.encoder(inputs[:,:,:,0:2], is_training=is_training)
        b_list = b_list[0:-1]
        b_list.reverse()
        x1 = self.decoder.sample(x, b_list, is_training=is_training)
        x1 = self.conv(x1)
        mask = inputs[:, :, :, 2:3]
        x1 = tf.keras.activations.sigmoid(x1)
        masks = self.get_gaussian_pyramid(mask, 4)
        res_gaus = self.get_gaussian_pyramid(x1, 4)
        inp_gaus = self.get_gaussian_pyramid(inputs[:, :, :, 0:1], 4)
        res_ = self.get_laplacian_pyramid(res_gaus)
        inp_ = self.get_laplacian_pyramid(inp_gaus)
        x1 = self.build_from_pyramid(inp_, res_, masks)
        return x1
