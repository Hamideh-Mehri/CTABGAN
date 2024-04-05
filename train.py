import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import numpy as np


class Train(object):
    """
    generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300."""
            
    def __init__(self, transformer, sampler, generator, gside, discriminator, discriminator_rep, dside, classifier, random_dim=100, batch_size=500, learning_rate=2e-4, weight_decay=1e-5
                ,discriminator_steps=1,log_frequency=True, verbose=True, epochs=5):
        
        self._lr =learning_rate
        self.l2scale = weight_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._random_dim = random_dim
        self._transformer = transformer
        self._sampler = sampler
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_rep = discriminator_rep
        self.classifier = classifier
        self.gside = gside
        self.dside = dside

    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        """Samples from the Gumbel-Softmax distribution
        :cite:`maddison2016concrete`, :cite:`jang2016categorical` and
        optionally discretizes.
        Parameters
        ----------
        logits: tf.Tensor
            Un-normalized log probabilities.
        tau: float, default=1.0
            Non-negative scalar temperature.
        hard: bool, default=False
            If ``True``, the returned samples will be discretized as
            one-hot vectors, but will be differentiated as soft samples.
        dim: int, default=1
            The dimension along which softmax will be computed.
        Returns
        -------
        tf.Tensor
            Sampled tensor of same shape as ``logits`` from the
            Gumbel-Softmax distribution. If ``hard=True``, the returned samples
            will be one-hot, otherwise they will be probability distributions
            that sum to 1 across ``dim``.
        """

        gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
        gumbels = gumbel_dist.sample(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        output = tf.nn.softmax(gumbels, dim)

        if hard:
            index = tf.math.reduce_max(output, 1, keepdims=True)
            output_hard = tf.cast(tf.equal(output, index), output.dtype)
            output = tf.stop_gradient(output_hard - output) + output
        return output
        
    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(tf.math.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return tf.concat(data_t, axis=1)

    def cross_entropy_conditional_loss(self, data, convec, mask, output_info):
        """
        Used to compute the conditional loss for ensuring the generator produces the desired category as specified by the conditional vector
        Inputs:
        1) data -> raw data synthesized by the generator 
        2) output_info -> column informtion corresponding to the data transformer
        3) convec -> conditional vectors used to synthesize a batch of data
        4) mask -> a matrix to identify chosen one-hot-encodings across the batch
        Outputs:
        1) loss -> conditional loss corresponding to the generated batch 
        """
        tmp_loss = []
        st = 0
        st_c = 0
        output_info_flat = [elem for sublist in output_info for elem in sublist]
        for column_info in output_info_flat:
            if column_info.activation_fn == 'tanh':
                st += column_info.dim
                continue
            elif column_info.activation_fn == 'softmax':
                ed = st + column_info.dim
                ed_c = st_c + column_info.dim
                logits = data[:, st:ed]
                labels = convec[:, st_c:ed_c]
                tmp = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
                tmp_loss.append(tmp)
                st = ed
                st_c = ed_c
        tmp_loss = tf.stack(tmp_loss, axis=1)
        loss = tf.reduce_mean(tmp_loss * mask)  
        return loss      

    def _convert_to_image(self, data):
        if self.dside * self.dside > len(data[0]):
            # Tabular data records are padded with 0 to conform to square shaped images
            padding = tf.zeros((len(data), self.dside * self.dside - len(data[0])), dtype=data.dtype)
            data = tf.concat([data, padding], axis=1)

        reshaped_data = tf.reshape(data, (-1, self.dside, self.dside, 1))
        return reshaped_data

    def train(self, raw_data):
        optimizer_params = dict(learning_rate = self._lr, beta_1=0.5, beta_2=0.9, epsilon=1e-3, decay=self.l2scale)

        optimizerG = tf.keras.optimizers.Adam(**optimizer_params)
        optimizerD = tf.keras.optimizers.Adam(**optimizer_params)
        optimizerC = tf.keras.optimizers.Adam(**optimizer_params)
        
        steps_per_epoch = max(len(raw_data)// self._batch_size, 1)

        for i in tqdm(range(self._epochs)):
            for id_ in range(steps_per_epoch):
                # Sampling noise vectors using a standard normal distribution
                noisez = tf.random.normal([self._batch_size, self._random_dim])
                #sampling conditional vector
                vec, mask, idx, opt1prime = self._sampler.sample_condvec_train(self._batch_size)
                vec = tf.convert_to_tensor(vec, dtype=tf.float32)
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                # Concatenating conditional vectors and converting resulting noise vectors into the image domain to be fed to the generator as input
                noisez = tf.concat([noisez, vec], axis=1)
                noisez = tf.reshape(noisez, [self._batch_size, 1, 1, self._random_dim + self._sampler.n_opt])

                # Sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator
                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)
                real = self._sampler.sample_data(self._batch_size, idx[perm], opt1prime[perm])
                real = tf.convert_to_tensor(real.astype('float32'))

                # Storing shuffled ordering of the conditional vectors
                vec_perm = tf.gather(vec, perm)
                # generating synthetic data as an image
                fake = self.generator(noisez)
                # converting it into the tabular domain as per format of the trasformed training data
                faket = tf.reshape(fake, (-1, self.gside * self.gside))
                # applying final activation on the generated data (i.e., tanh for numeric and gumbel-softmax for categorical)
                fakeact = self._apply_activate(faket)
                # the generated data is then concatenated with the corresponding condition vectors
                fake_cat = tf.concat([fakeact, vec], axis=1)
                # the real data is also similarly concatenated with corresponding conditional vectors   
                real_cat = tf.concat([real, vec_perm], axis=1)
                # transforming the real and synthetic data into the image domain for feeding it to the discriminator
                real_cat_d = self._convert_to_image(real_cat)
                fake_cat_d = self._convert_to_image(fake_cat)

                #update discriminator weights
                with tf.GradientTape() as disc_tape:
                    # computing the probability of the discriminator to correctly classify real samples hence y_real should ideally be close to 1
                    y_real = self.discriminator(real_cat_d)
                    # computing the probability of the discriminator to correctly classify fake samples hence y_fake should ideally be close to 0
                    y_fake = self.discriminator(fake_cat_d)
                    loss_D_real = tf.keras.losses.binary_crossentropy(tf.ones_like(y_real), y_real)
                    loss_D_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(y_fake), y_fake)
                    disc_loss = loss_D_real + loss_D_fake
                grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                optimizerD.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

                # similarly sample noise vectors and conditional vectors
                # input to generator(this time for training generator)
                noisez = tf.random.normal([self._batch_size, self._random_dim])
                #sampling conditional vector
                vec, mask, idx, opt1prime = self._sampler.sample_condvec_train(self._batch_size)
                vec = tf.convert_to_tensor(vec, dtype=tf.float32)
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                # Concatenating conditional vectors and converting resulting noise vectors into the image domain to be fed to the generator as input
                noisez = tf.concat([noisez, vec], axis=1)
                noisez = tf.reshape(noisez, [self._batch_size, 1, 1, self._random_dim + self._sampler.n_opt])
                
                #update generator weights
                with tf.GradientTape() as gen_tape, tf.GradientTape() as info_tape:
                    fake = self.generator(noisez)
                    faket = tf.reshape(fake, (-1, self.gside * self.gside))
                    fakeact = self._apply_activate(faket)
                    fake_cat = tf.concat([fakeact, vec], axis=1)
                    fake_cat = self._convert_to_image(fake_cat)
                    y_fake = self.discriminator(fake_cat)
                    # computing the conditional loss to ensure the generator generates data records with the chosen category as per the conditional vector
                    cross_entropy = self.cross_entropy_conditional_loss(faket, vec, mask, self._transformer.output_info_list)
                    # computing the loss to train the generator where we want y_fake to be close to 1 to fool the discriminator 
                    # and cross_entropy to be close to 0 to ensure generator's output matches the conditional vector
                    gen_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(y_fake), y_fake) + cross_entropy

                    info_fake = self.discriminator_rep(fake_cat)
                    info_real = self.discriminator_rep(real_cat_d)
                    # Computing the information loss by comparing means and stds of real/fake feature representations extracted from discriminator's penultimate layer
                    loss_mean = tf.norm(tf.math.reduce_mean(info_fake, axis=0) - tf.math.reduce_mean(info_real, axis=0), ord=1)
                    loss_std = tf.norm(tf.math.reduce_std(info_fake, axis=0) - tf.math.reduce_std(info_real, axis=0), ord=1)
                    loss_info = loss_mean + loss_std

                grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                grads_info = info_tape.gradient(loss_info, self.generator.trainable_variables)
                gradients_of_generator = [g+info for g, info in zip(grads_gen, grads_info)]
                optimizerG.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                # The classifier module is used in case there is a target column associated with ML tasks
                print('yes')
                
                                