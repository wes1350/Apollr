import tensorflow as tf
import numpy as np
import os, re, time
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

DATA_PATH = "test_id_playcount_pairs.npz"
CHECK_POINT_DIR = "checkpoints"
REG = 0.0

class CFNet:

    def __init__(self, latent_dim):
         
        with np.load(DATA_PATH) as data:
            self.data = data["arr_0"]
            
        self.rnd_range = len(self.data) // 2 - 1
        self.max_song_index = self.data[-1][0] - 1
        self.latent_dim = latent_dim
        self.reuse = False
        self.leak = 0.2

    @staticmethod
    def set_up_folders(output_folder, model_number):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        while os.path.exists(os.path.join(output_folder, model_number)):
            curr_num = re.sub('.*?([0-9]*)$', r'\1', model_number)
            number = int(curr_num) + 1
            model_number = model_number[:-len(curr_num)] + str(number)

        os.mkdir(os.path.join(output_folder, model_number))
        return model_number
    
    @staticmethod
    def l_relu(x, leak):
        return tf.maximum(x, x * leak, name="leaky_relu")

    def create_batch(self, batch_size):
        #samples = np.random.randint(0, self.rnd_range, batch_size)
        batch_size = len(self.data)//2 - 1
        samples = np.arange(0, batch_size)
        base = 2*samples
        song_indices = self.data[base]
        play_counts = self.data[base + 1]
        play_counts = np.hstack(play_counts)
        arrays = []
        for i in range(len(song_indices)):
            n = len(song_indices[i])
            pairs = np.array([i]*n)
            arrays.append(np.dstack((pairs, song_indices[i]))[0])
        
        
        indices = np.vstack(arrays)    
        return tf.SparseTensorValue(indices=indices, values=play_counts, dense_shape=[batch_size, self.max_song_index])

    def predict(self, inp, scale=REG):

        with tf.variable_scope('net', reuse=self.reuse):
            with tf.variable_scope('encode'):
                w = tf.get_variable("w1", shape=[self.max_song_index, self.latent_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b1", initializer=tf.constant(0.1, shape=[self.latent_dim]))

                layer = tf.nn.bias_add(tf.sparse_tensor_dense_matmul(inp, w), b)
                layer = tf.nn.elu(layer)

            with tf.variable_scope('decode'):
                w = tf.get_variable("w2", shape=[self.latent_dim, self.max_song_index],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b2", initializer=tf.constant(0.1, shape=[self.max_song_index]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)
                layer = tf.nn.relu(layer)

        self.reuse = True
        return layer
    
    
    def deep_predict(self, inp, scale=REG, l_dim_scale=3):

        with tf.variable_scope('net', reuse=self.reuse):
            with tf.variable_scope('encode'):
                w = tf.get_variable("w1", shape=[self.max_song_index, self.latent_dim*l_dim_scale],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b1", initializer=tf.constant(0.1, shape=[self.latent_dim*l_dim_scale]))

                layer = tf.nn.bias_add(tf.sparse_tensor_dense_matmul(inp, w), b)
                layer = tf.nn.elu(layer)
                
            with tf.variable_scope('h1'):
                w = tf.get_variable("w", shape=[self.latent_dim*l_dim_scale, self.latent_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[self.latent_dim]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)
                layer = tf.nn.elu(layer)
            
            with tf.variable_scope('h2'):
                w1 = tf.get_variable("w", shape=[self.latent_dim, self.latent_dim*l_dim_scale],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b1 = tf.get_variable("b", initializer=tf.constant(0.1, shape=[self.latent_dim*l_dim_scale]))

                layer = tf.nn.bias_add(tf.matmul(layer, w1), b1)
                layer = tf.nn.elu(layer)

            with tf.variable_scope('decode'):
                w = tf.get_variable("w", shape=[self.latent_dim*l_dim_scale, self.max_song_index],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[self.max_song_index]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)
                layer = tf.nn.relu(layer)

        self.reuse = True
        return layer

    def train(self, train_steps=1000, learning_rate=0.001, batch_size=128, print_per=20, save_per=1000, deep=False):
        
        pred = self.deep_predict if deep else self.predict

        output_folder = os.path.join(CHECK_POINT_DIR, "")
        model_name = CFNet.set_up_folders(output_folder, model_number="model1")

        sparse_vec = tf.sparse_placeholder(tf.float32)
        predictions = pred(sparse_vec)
        
        filter_values = tf.ones_like(sparse_vec.values)        
        filter_vec = tf.SparseTensor(sparse_vec.indices, filter_values, sparse_vec.dense_shape)
        
        predictions = filter_vec * predictions
        negative_sparse_vec = sparse_vec*-1
        se = tf.square(tf.sparse_add(predictions, negative_sparse_vec))
        se = tf.sparse_reduce_sum(se)
        
        nnz = tf.sparse_reduce_sum(filter_vec)
        reg_w = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_w = tf.reduce_sum(reg_w)
        se = se/nnz 
        
        trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(se)

        saver = tf.train.Saver()
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            for step in range(1, train_steps + 1):

                _, loss = sess.run([trainer, se], feed_dict={sparse_vec: self.create_batch(batch_size)})
                losses.append(loss)
                if step % print_per == 0:
                    print("loss={}, on update step {}".format(loss, step))
                    end = time.time()
                    print("Average time for 1 iteration was {}s".format((end - start) / print_per))
                    start = time.time()

                if step % save_per == 0 or step == train_steps:
                    saver.save(sess, "{}/{}/".format(output_folder, model_name), step)

        return losses


def save_plot(filename, title, x_label, y_label, x_data, y_data):

    plt.plot(x_data, y_data)
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format="png")

if __name__ == "__main__":
    net = CFNet(latent_dim=100)
    losses = net.train(deep=True)
    save_plot("losses_deep", "loss vs. train iter", "train_step", "loss", list(range(len(losses))), losses)

