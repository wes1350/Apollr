import tensorflow as tf
import numpy as np
import os, re, time
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

DATA_PATH = os.path.join("data", "gaussian_ratings.npz")
CHECK_POINT_DIR = "checkpoints"
REG = 0.0
PRINT_EVERY = 100
KEEP_PROB = 0.8

class CFNet:
    def __init__(self, latent_dim, training=True):

        with np.load(DATA_PATH, mmap_mode='r') as data:
            self.data = data["train"]
            self.val = data["val"]

        self.max_song_index = self.data[-1][0]
        self.latent_dim = latent_dim
        self.reuse = False
        self.training = training

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
    def l_relu(x, leak=0.2):
        return tf.maximum(x, x * leak, name="leaky_relu")

    def create_batch(self, batch_size, val=False, test=False):        
        if test:
            batch_size = len(self.test) // 2 - 1
            samples = np.arange(0, batch_size)
            data = self.test
        else:
            data = self.val if val else self.data
            samples = np.random.randint(0, len(data) // 2 - 1, batch_size)
        
        base = 2 * samples
        song_indices = data[base]
        play_counts = data[base + 1]
        play_counts = np.hstack(play_counts)
        arrays = []
        for i in range(len(song_indices)):
            n = len(song_indices[i])
            pairs = np.array([i] * n)
            arrays.append(np.dstack((pairs, song_indices[i]))[0])

        indices = np.vstack(arrays)
        return tf.SparseTensorValue(indices=indices, values=play_counts, dense_shape=[batch_size, self.max_song_index])

    def predict(self, inp, keep, scale=REG):
        
        with tf.variable_scope('net', reuse=self.reuse):
            with tf.variable_scope('encode'):
                w = tf.get_variable("w1", shape=[self.max_song_index, self.latent_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[self.latent_dim]))

                layer = tf.nn.bias_add(tf.sparse_tensor_dense_matmul(inp, w), b)
                layer = tf.nn.dropout(layer, keep_prob=keep)
                layer = tf.layers.batch_normalization(layer, training=self.training, name="sbatch1")
                layer = CFNet.l_relu(layer)

            with tf.variable_scope('decode'):
                w = tf.get_variable("w2", shape=[self.latent_dim, self.max_song_index],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b2", initializer=tf.constant(0.0, shape=[self.max_song_index]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)

        self.reuse = True
        return layer

    def deep_predict(self, inp, keep, scale=REG, l_dim_scale=2):

        with tf.variable_scope('net', reuse=self.reuse):
            with tf.variable_scope('encode'):
                w = tf.get_variable("w1", shape=[self.max_song_index, self.latent_dim * l_dim_scale],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[self.latent_dim * l_dim_scale]))
 
                layer = tf.nn.bias_add(tf.sparse_tensor_dense_matmul(inp, w), b)
                layer = tf.nn.dropout(layer, keep_prob=keep)
                layer = tf.layers.batch_normalization(layer, training=self.training, name="dbatch1")
                layer = CFNet.l_relu(layer)


            with tf.variable_scope('h1'):
                w = tf.get_variable("w", shape=[self.latent_dim * l_dim_scale, self.latent_dim],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[self.latent_dim]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)
                layer = tf.nn.dropout(layer, keep_prob=keep)
                layer = tf.layers.batch_normalization(layer, training=self.training, name="dbatch2")
                layer = CFNet.l_relu(layer)


            with tf.variable_scope('h2'):
                w = tf.get_variable("w", shape=[self.latent_dim, self.latent_dim * l_dim_scale],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[self.latent_dim * l_dim_scale]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)
                layer = tf.nn.dropout(layer, keep_prob=keep)
                layer = tf.layers.batch_normalization(layer, training=self.training, name="dbatch3")
                layer = CFNet.l_relu(layer)

            with tf.variable_scope('decode'):
                w = tf.get_variable("w", shape=[self.latent_dim * l_dim_scale, self.max_song_index],
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=tf.contrib.layers.l2_regularizer(scale))
                b = tf.get_variable("b", initializer=tf.constant(0.0, shape=[self.max_song_index]))

                layer = tf.nn.bias_add(tf.matmul(layer, w), b)

        self.reuse = True
        return layer

    def train(self, train_steps=5100, learning_rate=0.001, batch_size=128, print_per=PRINT_EVERY, save_per=100, deep=False):

        pred = self.deep_predict if deep else self.predict

        output_folder = os.path.join(CHECK_POINT_DIR, "")
        model_name = CFNet.set_up_folders(output_folder, model_number="model1")
        
        self.checkpoint = "{}/{}/".format(output_folder, model_name)

        sparse_vec = tf.sparse_placeholder(tf.float32, name="sparse")
        dropout = tf.placeholder(tf.float32, name='dropout')
        predictions = pred(sparse_vec, dropout)
        predictions = tf.identity(predictions, name="pred")

        filter_values = tf.ones_like(sparse_vec.values)
        filter_vec = tf.SparseTensor(sparse_vec.indices, filter_values, sparse_vec.dense_shape)
        negative_sparse_vec = sparse_vec * -1
        se = tf.square(tf.sparse_add(predictions, negative_sparse_vec))
        se = filter_vec * se
        se = tf.sparse_reduce_sum(se)

        nnz = tf.sparse_reduce_sum(filter_vec)
        reg_w = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_w = tf.reduce_sum(reg_w)
        se = se / nnz + reg_w
        
        indices = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="indices")
        counts = tf.placeholder(dtype=tf.float32, shape=[None], name="counts")
        
        error_percent = tf.placeholder(dtype=tf.float32, shape=None, name="tolerance")
        accuracies = tf.abs(tf.gather_nd(predictions, indices) - counts) <= tf.multiply(error_percent, counts)
        accuracies = tf.reduce_mean(tf.cast(accuracies, tf.float32), name="accuracy")
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(se)

        saver = tf.train.Saver()
        losses = []
        val_losses = []
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            for step in range(1, train_steps + 1):

                _, loss = sess.run([trainer, se], feed_dict={sparse_vec: self.create_batch(batch_size),
                                                             dropout: KEEP_PROB})
                losses.append(loss)
                v_loss = sess.run(se, feed_dict={sparse_vec: self.create_batch(batch_size, val=True),
                                                 dropout: KEEP_PROB})
                val_losses.append(v_loss)
                if step % print_per == 0:
                    print("loss={}, on update step {}".format(loss, step))
                    end = time.time()
                    print("Average time for 1 iteration was {}s".format((end - start) / print_per))
                    
                    print("Current Validation loss={}".format(v_loss))
                    start = time.time()

                if step % save_per == 0:
                    saver.save(sess, "{}/{}/".format(output_folder, model_name), step)
            
            saver.save(sess, "{}/{}/final".format(output_folder, model_name))
            loss = sess.run(se, feed_dict={sparse_vec: self.create_batch(batch_size, val=True), 
                                           dropout: KEEP_PROB})
            val_losses.append(loss)
        return losses, val_losses
    
    
    def evaluate(self, deep=False): 
        del self.data
        del self.val
        self.training=False
        with np.load(DATA_PATH, mmap_mode='r') as data:
            self.test = data['test'][:4000]
        
        test_batch = self.create_batch(-1, test=True)
        del self.test
        
        indx = []
        play_c = []
        
        prev_user = test_batch.indices[0]
        
        for i in range(1, len(test_batch.indices)):
            current_user = test_batch.indices[i]
            
            if prev_user[0] != current_user[0]: #new user
                prev_user = current_user
                if i > 0:
                    indx.append(test_batch.indices[i-1])
                    play_c.append(test_batch.values[i-1])
                    test_batch.values[i-1] = 0
        
        indx.append(test_batch.indices[i-1])
        play_c.append(test_batch.values[i-1])
        test_batch.values[i-1] = 0
        indx = np.array(indx)
        play_c = np.array(play_c, dtype=np.float32)
        
        meta = tf.train.import_meta_graph(self.checkpoint + "//final.meta")
        
        with tf.Session() as sess:
            meta.restore(sess, tf.train.latest_checkpoint(self.checkpoint))
            graph = tf.get_default_graph()
            accuracies = graph.get_operation_by_name("accuracy").outputs[0]
            indices = graph.get_tensor_by_name("indices:0")
            counts = graph.get_tensor_by_name("counts:0")
            s_idxs = graph.get_tensor_by_name("sparse/indices:0")
            s_vals = graph.get_tensor_by_name("sparse/values:0")
            s_shape = graph.get_tensor_by_name("sparse/shape:0")
            e_per = graph.get_tensor_by_name("tolerance:0")
            dropout = graph.get_tensor_by_name("dropout:0")
            
            accs = sess.run(accuracies, feed_dict={s_idxs: np.array(test_batch.indices), 
                                                   s_vals: np.array(test_batch.values),
                                                   s_shape: np.array(test_batch.dense_shape),
                                                   indices: indx , counts: play_c, e_per:0.1,
                                                   dropout:1.0})
            print("Accuracy on the test set is: ", accs)
        


def save_plot(filename, title, x_label, y_label, x_train, y_train, x_val, y_val):
    plt.plot(x_train, y_train, '-b', label='Training loss')
    plt.plot(x_val, y_val, '-r', label='Val loss')
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.legend(loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format="png")

deep = False
if __name__ == "__main__":
    net = CFNet(latent_dim=64)
    losses, val_losses = net.train(deep=deep)
    save_plot("losses.png", "Loss vs Train Iteration", "train_step", "loss", list(range(len(losses)-100)), 
              losses[100:], list(range(len(val_losses)-100)), val_losses[100:])
#    net.checkpoint= "checkpoints/model8"
    net.evaluate(deep=deep)
