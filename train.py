import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
import time
from datetime import datetime
from scipy.misc import imread, imsave
import matplotlib as mpl
import sys
from skimage import measure

mpl.use('Agg')
import matplotlib.pyplot as plt


def create_cnn_layer(inputs, n_filters, kernel_size, padding='same', activation=tf.nn.relu, dropout=None):
    cnn = tf.layers.conv2d(inputs=inputs,
                            filters=n_filters,
                            kernel_size=[kernel_size, kernel_size],
                            padding=padding,
                            activation=activation,
                            )
    return tf.nn.dropout(cnn, keep_prob=1.0)


class Model(object):
    def __init__(self, run_name=None):
        x_dim_train = 10
        y_dim_train = 10
        x_dim_target = 10
        y_dim_target = 10
        kernel_sizes = [19, 1, 1, 1, 1, 3, 1, 5, 5, 5, 3, 5, 5, 1, 7, 7, 9]
        layers = 20
        kernel_sizes = [15, 10, 6, 2, 1, 2, 6, 10]
        kernel_sizes = [5] * layers
        # kernel_sizes = [9, 1, 5]
        channel_sizes = [128, 320, 320, 320, 128, 128, 512, 320, 128, 128, 128, 128, 128, 256, 64, 3]
        channel_sizes = [64] * layers
        # channel_sizes = [64, 32, 3]
        self.layers = min(len(kernel_sizes), len(channel_sizes))

        self.input = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.dropout = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.network = [create_cnn_layer(self.input, channel_sizes[0], kernel_sizes[0], dropout=self.dropout)]

        for i in range(1, self.layers):
            next_layer = create_cnn_layer(self.network[i-1], channel_sizes[i], kernel_sizes[i], dropout=self.dropout)
            next_layer = tf.add(next_layer, self.network[i-1])
            self.network.append(next_layer)

        self.predicted_image = create_cnn_layer(self.network[-1], 3, 1, activation=tf.identity, dropout=self.dropout)
        # long term skip connection
        # self.predicted_image = tf.add(self.predicted_image, self.input)

        self.target_image = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.loss = tf.nn.l2_loss(tf.subtract(self.target_image, self.predicted_image))

        starter_learning_rate = 1e-5
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, \
                                                        3000, 1., staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.grads = tf.gradients(self.loss, tf.trainable_variables())
        self.capped_gradients, _ = tf.clip_by_global_norm(self.grads, clip_norm=10)
        self.grads_and_vars = list(zip(self.capped_gradients, tf.trainable_variables()))
        self.optimizer = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss, global_step=self.global_step)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        
        self.sess = tf.Session(config=config)

        self.init = tf.global_variables_initializer()
        self.loss_summary = tf.summary.scalar('train/loss', self.loss)
        self.lr_summary = tf.summary.scalar('train/learning_rate', self.learning_rate)
        self.validation_loss_summary = tf.summary.scalar('validation/validation_loss', self.loss)
        train_summaries = [self.loss_summary, self.lr_summary]
        for grad, var in self.grads_and_vars:
                print(var.name)
                # train_summaries.append(tf.summary.histogram("gradients/" + var.name, grad))
        self.train_summaries = tf.summary.merge(train_summaries)
        self.validation_summaries = tf.summary.merge([self.validation_loss_summary])
       

        if run_name is None:
            datestring = datetime.strftime(datetime.now(), '%m-%d_%H%M%S')
            self.run_name = datestring
        else:
            self.run_name = run_name

        log_dir = "./logs/" + self.run_name + "/"
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=15)

    def fit(self, restore_path=None):
        train_data_path = "./Data/Cropped/Blurred/Training/"
        target_data_path = "./Data/Cropped/Clear/"
        validation_data_path = "./Data/Cropped/Blurred/Validation/"
        train_data_files = []
        target_data_files = []
        validation_data_files = []
        valdation_target_data_files = []

        for root, dirs, files in os.walk(train_data_path):
            for f in files:
                if f[-3:] in ["jpg", "png"]:
                    train_data_files.append(os.path.join(train_data_path, f[:8], f))
                    target_data_files.append(os.path.join(target_data_path, f[:8], f))

        for root, dirs, files in os.walk(validation_data_path):
            for f in files:
                if f[-3:] in ["jpg", "png"]:
                    validation_data_files.append(os.path.join(validation_data_path, f[:8], f))
                    valdation_target_data_files.append(os.path.join(target_data_path, f[:8], f))

        all_data_files = np.array(list(zip(train_data_files, target_data_files)))
        
        np.random.shuffle(all_data_files)
        all_validation_files = np.array(list(zip(validation_data_files, valdation_target_data_files)))

        batch_size = 1
        validation_batch_size = 1

        num_batches = int(np.ceil(all_data_files.shape[0] * 1. / batch_size))
        num_validation_batches = int(np.ceil(all_validation_files.shape[0] * 1. / validation_batch_size))
        all_data_files = np.array_split(all_data_files, num_batches)
        all_validation_files = np.array_split(all_validation_files, num_validation_batches)

        num_epochs = 10000

        if restore_path is None:
            print("initialized variables")
            self.sess.run(self.init)
        else:
            print("model restored")
            self.saver.restore(self.sess, restore_path)

        for epoch in range(num_epochs):
            np.random.shuffle(all_data_files)
            self.predict()
            for batch_number, batch in enumerate(all_data_files):
                if batch_number % (int(num_batches / 2)) == 0 and epoch + batch_number != 0:
                    validation_loss = 0

                    for validation_batch_number, validation_batch in enumerate(all_validation_files):
                        # print("Validation {} / {}".format(validation_batch_number, num_validation_batches))

                        validation_data = []
                        validation_target_data = []

                        for i, f in enumerate(validation_batch):
                            validation_img = imread(f[0]) * 1. / 255
                            target_validation_img = imread(f[1]) * 1. / 255
                            validation_data.append(validation_img) 
                            validation_target_data.append(target_validation_img)

                        # img = self.sess.run(self.predicted_image, feed_dict={self.input: validation_data})
                        # imsave(os.path.join("./Data/Predictions/Validation/", f[0][-22:]), img[0])
                        img, val_loss = self.sess.run([self.predicted_image, self.loss],
                                                         feed_dict={self.input: np.array(validation_data),
                                                                    self.target_image: np.array(validation_target_data),
                                                                    self.dropout: 1.0})
                        img = img.clip(min=0, max=1)
                        imsave(os.path.join("./Data/Predictions/Validation/", f[0][-22:]), 255 * np.vstack([validation_data[0], img[0], validation_target_data[0]]))
                        validation_loss += val_loss

                    validation_loss /= num_validation_batches
                    print("Validation Loss:", validation_loss, " " * 20)

                    validation_summary = self.sess.run(self.validation_summaries,
                                                       feed_dict={self.loss: validation_loss})
                    self.summary_writer.add_summary(validation_summary, step)
                    self.summary_writer.flush()
                    datestring = datetime.strftime(datetime.now(), '%m-%d_%H%M%S')
                    save_path = "./models/" + datestring + ".ckpt"
                    self.saver.save(self.sess, save_path)

                duration = time.time()
                data = np.empty(1, dtype=np.float32)
                target_data = np.empty(1, dtype=np.float32)

                for i, f in enumerate(batch):
                    train_img = imread(f[0])* 1. / 255
                    target_img = imread(f[1]) * 1. / 255
                    data = np.array([train_img])
                    target_data = np.array([target_img])

                summaries, _, step, loss = self.sess.run(
                    [self.train_summaries, self.optimizer, self.global_step, self.loss],
                    feed_dict={self.input: data, self.target_image: target_data, self.dropout: 0.8})
                duration = time.time() - duration

                self.summary_writer.add_summary(summaries, step)

                print("Epoch {} of {}: Batch {} of {}: Loss: {:.5f}, Duration: {:.2f}sec\r".format(
                    epoch, num_epochs, batch_number, num_batches, loss, duration), end="\r")

    def predict(self, restore_path=None, test_data_dir="./Data/Test_Data/",
                prediction_dir="./Data/Predictions/"):
        if restore_path is not None:
            self.saver.restore(self.sess, restore_path)

        for f in os.listdir(test_data_dir):
            if f.endswith(".jpg") or f.endswith(".png"):
                img = np.array([imread(os.path.join(test_data_dir, f)) * 1. / 255])

                prediction = self.sess.run(self.predicted_image, feed_dict={self.input: img, self.dropout: 1.0})
                prediction = prediction.clip(min=0, max=1)
                imsave("./Data/Predictions/Test/" + f, prediction[0] * 255.)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        m = Model(run_name=sys.argv[2])
        m.fit(restore_path="models/" + sys.argv[1])
    else:
        m = Model(run_name=None)
        m.fit(restore_path=None)
    # m.predict("./models/10-29_195049.ckpt")
