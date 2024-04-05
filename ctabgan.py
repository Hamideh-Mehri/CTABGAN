import tensorflow as tf
import math

class ConvTwo(tf.keras.layers.Layer):
    def __init__(self, weight_shape, bias_shape, stridesList, padding, use_bias=True):
        super(ConvTwo,self).__init__()
        self.w = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)(shape=weight_shape), trainable = True)
        self.b = tf.Variable(tf.keras.initializers.zeros()(shape=bias_shape), trainable = True)
        self.strides = stridesList
        self.padding = padding
        self.use_bias = use_bias
    def call(self, inputs):
        w = self.w 
        b = self.b 
        if self.use_bias:
           return tf.nn.conv2d(inputs, w, self.strides, self.padding) + b
        else:
           return tf.nn.conv2d(inputs, w, self.strides, self.padding)

class Dense(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
     super(Dense, self).__init__()
     bound = 1/math.sqrt(input_dim)
     w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
     b_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
  def call(self,inputs):
      return tf.matmul(inputs, self.w) + self.b


class ConvTwoTranspose(tf.keras.layers.Layer):
    def __init__(self, weight_shape, outputshape, stridesList, padding):
        super(ConvTwoTranspose,self).__init__()
        self.w = tf.Variable(tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)(shape=weight_shape), trainable = True)
        self.outputshape = outputshape
        self.strides = stridesList
        self.padding = padding

    def call(self, inputs):
        w = self.w 
        return tf.nn.conv2d_transpose(inputs, w, self.outputshape, self.strides, self.padding) 


class CTABGAN(object):
    def __init__(self, random_dim = 100, num_channels = 64, classifier_dim = (256, 256, 256, 256)):
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.classifier_dim = classifier_dim




    def make_generator(self, sampler, transformer, batchSize):
        data_dim = transformer.output_dimensions		
        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                gside = i
                break

       
        # computing the dimensionality of hidden layers
        layer_dims = [(1, gside), (self.num_channels, gside // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        inp = tf.keras.layers.Input(shape=(1,1, self.random_dim + sampler.n_opt))
        # inputs to ConvTwoTranspose: weight_shape, outshape, strideslist, padding
        x = ConvTwoTranspose((layer_dims[-1][1], layer_dims[-1][1], layer_dims[-1][0], self.random_dim + sampler.n_opt), [batchSize, 2, 2, layer_dims[-1][0]],
                               [1,1,1,1], 'VALID')(inp)
        num_filter_prev = layer_dims[-1][0]
        i = 2
        for curr in reversed(layer_dims[:-1]):
            x = tf.keras.layers.BatchNormalization(axis=-1,gamma_initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                                           beta_initializer=tf.keras.initializers.Zeros())(x)
            x = tf.keras.layers.ReLU()(x)
            x = ConvTwoTranspose((4, 4, curr[0], num_filter_prev), [batchSize, 2**i, 2**i, curr[0]], [1,2,2,1], 'SAME')(x)
            i += 1
            num_filter_prev = curr[0]
        generator = tf.keras.models.Model(inp, x)
        return generator, gside
        # x = tf.keras.layers.Conv2DTranspose(filters=layer_dims[-1][0], kernel_size=layer_dims[-1][1], strides=1, padding='valid', use_bias=False)(inp)
        # for curr in reversed(layer_dims[:-1]):
        #     x = tf.keras.layers.BatchNormalization()(x)
        #     x = tf.keras.layers.ReLU()(x)
        #     x = tf.keras.layers.Conv2DTranspose(filters=curr[0], kernel_size=4, strides=2, padding='same', use_bias=True)(x)
        # generator = tf.keras.models.Model(inp, x)








    def make_discriminator(self, sampler, transformer):
        # obtaining the desired height/width for converting tabular data records to square images for feeding it to discriminator network 		
        sides = [4, 8, 16, 24, 32]
        data_dim = transformer.output_dimensions		
        # the discriminator takes the transformed training data concatenated by the corresponding conditional vectors as input
        col_size_d = data_dim + sampler.n_opt
        for i in sides:
            if i * i >= col_size_d:
                dside = i
                break

        layer_dims = [(1, dside), (self.num_channels, dside // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            # the number of channels increases by a factor of 2 whereas the height/width decreases by the same factor with each layer
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        inp_disc = tf.keras.layers.Input(shape=(dside, dside, 1))
        x = inp_disc
        kernel_depth = 1
        for curr in layer_dims[1:]:
            kernel_size = 4
            num_filters = curr[0]
            stridesList = [1,2,2,1]
            padding = 'SAME'
            weight_shape = (kernel_size, kernel_size, kernel_depth, num_filters)
            bias_shape = (num_filters,)
            x = ConvTwo(weight_shape, bias_shape, stridesList, padding)(x)
            x = tf.keras.layers.BatchNormalization(axis=-1,gamma_initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02), 
                                                           beta_initializer=tf.keras.initializers.Zeros())(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            kernel_depth = num_filters

        feature_rep = x

        kernel_size=layer_dims[-1][1]
        num_filters = 1
        stridesList = [1,1,1,1]
        padding ='VALID'
        weight_shape = (kernel_size, kernel_size, kernel_depth, num_filters)
        bias_shape = (num_filters,)
        x = ConvTwo(weight_shape, bias_shape, stridesList, padding)(x)
        x = tf.keras.layers.Flatten()(x)
        input_dim = x.shape[1]
        x = Dense(input_dim, 1)(x)
        out = tf.keras.activations.sigmoid(x)
        
        discriminator = tf.keras.models.Model(inp_disc, out)
        discriminator_rep = tf.keras.models.Model(inp_disc, feature_rep)

        return discriminator, discriminator_rep, dside


        # x = inp
        # for curr in layer_dims[1:]:
        #     x = tf.keras.layers.Conv2D(filters=curr[0], kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        #     x = tf.keras.layers.BatchNormalization()(x)
        #     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        # feature_rep = x
        # x = tf.keras.layers.Conv2D(filters=1, kernel_size=layer_dims[-1][1], strides=1, padding='valid')(x)
        # x = tf.keras.layers.Flatten()(x)
        # out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        # discriminator = tf.keras.models.Model(inp, out)
        # discriminator_rep = tf.keras.models.Model(inp, feature_rep)

    def make_classifier(self, transformer, st_ed):
        #get the dimension of transformed data
        output_info = transformer.output_info_list
        input_dim = 0
        for info in output_info:
            for item in info:
                input_dim += item.dim
        #dimension of input to classifier
        dim = input_dim - (st_ed[1] - st_ed[0])

        inp_classifier = tf.keras.layers.Input(shape=(dim,))
        input_dim = dim
        x = inp_classifier
        for output_dim in self.classifier_dim:
            x = Dense(input_dim, output_dim)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            input_dim = output_dim
        input_to_last_layer = output_dim
        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1] - st_ed[0]) == 2:
            x = Dense(input_to_last_layer, 1)(x)
            out = tf.keras.activations.sigmoid(x)
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class   
        else:
            out = Dense(input_to_last_layer, st_ed[1] - st_ed[0])(x)
        classifier = tf.keras.models.Model(inp_classifier, out)
        return classifier


