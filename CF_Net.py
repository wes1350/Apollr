import tensorflow as tf
import numpy as np
import os, re, time
import matplotlib.pyplot as plt

DATA_PATH = "id_playcount_pairs.npy"
CHECK_POINT_DIR = "checkpoints"

class CFNet:

    def __init__(self, latent_dim):
        self.data = np.load(DATA_PATH)
        self.rnd_range = len(self.data) // 2 - 1
        self.max_song_index = self.data[-1][0] - 1
        self.latent_dim = latent_dim
        self.reuse = False

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

    def create_batch(self, batch_size):
        samples = np.random.randint(0, self.rnd_range, batch_size)
        base = 2*samples
        song_indices = self.data[base]
        play_counts = self.data[base + 1]

        batch_data = np.zeros([batch_size, self.max_song_index])
        for i in range(batch_size): batch_data[i, song_indices[i]] = play_counts[i]
        return batch_data

    def predict(self, input):

        with tf.variable_scope('net', reuse=self.reuse):
            with tf.variable_scope('encode'):
                w1 = tf.get_variable("w1", shape=[self.max_song_index, self.latent_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                b1 = tf.get_variable("b1", initializer=tf.constant(0.1, shape=[self.latent_dim]))

                layer1 = tf.nn.bias_add(tf.matmul(input, w1), b1)
                layer1 = tf.nn.relu(layer1)

            with tf.variable_scope('decode'):
                w2 = tf.get_variable("w2", shape=[self.latent_dim, self.max_song_index],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                b2 = tf.get_variable("b2", initializer=tf.constant(0.1, shape=[self.max_song_index]))

                out = tf.nn.bias_add(tf.matmul(layer1, w2), b2)
                out = tf.nn.relu(out)

        self.reuse = True
        return out

    def train(self, train_steps=100000, learning_rate=0.002, batch_size=32, print_per=100, save_per=1000):

        output_folder = os.path.join(CHECK_POINT_DIR, "")
        model_name = CFNet.set_up_folders(output_folder, model_number="model1")

        sparse_vec = tf.placeholder(tf.float32, [None, self.max_song_index])
        predictions = self.predict(sparse_vec)
        filter_vec = sparse_vec > 0
        filter_vec = tf.cast(filter_vec, tf.float32)
        predictions = tf.multiply(predictions, filter_vec)
        mse = tf.losses.mean_squared_error(sparse_vec, predictions)

        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

        saver = tf.train.Saver()
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            for step in range(1, train_steps + 1):

                _, loss = sess.run([trainer, mse], feed_dict={sparse_vec: self.create_batch(batch_size)})
                losses.append(loss)
                print(step)
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
    net = CFNet(latent_dim=500)
    losses = net.train()
    save_plot("losses", "loss vs. train iter", "train_step", "loss", list(range(len(losses))), losses)

