import tensorflow as tf
import numpy as np
import math, os, time

MAX_RATING = 5.0
FEATURES = 64
DATA_PATH = os.path.join("data", "ratings.npz")
PRINT_PER = 1
TRAIN_STEPS = 100

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def create_batch(data):

    max_song_index = data[-1][0]
    batch_size = len(data) // 2 - 1
    samples = np.arange(0, batch_size)

    base = 2 * samples
    song_indices = data[base]
    play_counts = data[base + 1]
    play_counts = np.hstack(play_counts)
    play_counts = play_counts.astype(np.float32)
    arrays = []
    for i in range(len(song_indices)):
        n = len(song_indices[i])
        pairs = np.array([i] * n)
        arrays.append(np.dstack((pairs, song_indices[i]))[0])

    indices = np.vstack(arrays)
    return tf.SparseTensor(indices=indices, values=play_counts, dense_shape=[batch_size, max_song_index])


def low_rank_fact():
    reg = 0.02

    with np.load(DATA_PATH, mmap_mode='r') as data:
        mat_rep = create_batch(data["train"])
    
    with np.load("mats.npz") as m:
        u = m["users"]
        v = m["artists"]

    negative_rep = mat_rep*-1
    mat_shape = mat_rep.get_shape().as_list()
    u_matrix = tf.get_variable(name="u", initializer=tf.constant(u),
                               regularizer=tf.contrib.layers.l2_regularizer(reg))

    v_matrix = tf.get_variable(name="v", initializer=tf.constant(v),
                               regularizer=tf.contrib.layers.l2_regularizer(reg))


    indices = negative_rep.indices
    rows_needed = tf.expand_dims(indices[:,0], axis=-1)
    cols_needed = tf.expand_dims(indices[:,1], axis=-1)

    row_vectors = tf.gather_nd(u_matrix, rows_needed)
    col_vectors = tf.gather_nd(tf.transpose(v_matrix), cols_needed)
    
    values_needed = tf.reduce_sum(tf.multiply(row_vectors, col_vectors), axis=1)
    approx = tf.SparseTensor(indices=indices, values=values_needed, dense_shape=mat_shape)

    square = tf.sparse_reduce_sum(tf.square(tf.sparse_add(approx, negative_rep)))
    reg_w = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_w = tf.reduce_sum(reg_w)
    loss = square + reg_w

    u_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, momentum=0.5).minimize(loss, var_list=[u_matrix])
    v_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, momentum=0.5).minimize(loss, var_list=[v_matrix])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for step in range(TRAIN_STEPS):
            _ = sess.run(v_optimizer)
            _ = sess.run(u_optimizer)

            if step % PRINT_PER == 0 or step==TRAIN_STEPS-1:
                train_loss = sess.run(loss)
                print("loss={}, on update step {}".format(train_loss, step))
                end = time.time()
                print("Average time for 1 iteration was {}s".format((end - start) / PRINT_PER))
                start = time.time()

        artist_mat = sess.run(v_matrix)
        user_mat = sess.run(u_matrix)

        return artist_mat, user_mat

if __name__ == "__main__":
    artists, users = low_rank_fact()
    np.savez_compressed("mats", artists=artists, users=users)

