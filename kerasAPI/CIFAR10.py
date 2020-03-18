import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# [50k, 32, 32, 3]  [10k, 1]
(x, y), (x_test, y_test) = datasets.cifar10.load_data()

batchsz = 128
y = tf.squeeze(y)
# [10k, 1] -> [10k]
y_test = tf.squeeze(y_test)  
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)
print('datasets: ', x.shape, y.shape, x.min(), x.max())


db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inp_dim, training=None):
        res = inp_dim @ self.kernel + self.bias
        return res


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        inputs = tf.reshape(inputs, [-1, 32*32*3])
        res = self.fc1(inputs)
        res = tf.nn.relu(res)
        res = self.fc2(res)
        res = tf.nn.relu(res)
        res = self.fc3(res)
        res = tf.nn.relu(res)
        res = self.fc4(res)
        res = tf.nn.relu(res)
        res = self.fc5(res)
        return res


if __name__ == "__main__":
    # train
    network = MyModel()
    network.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    network.fit(
        db,
        epochs=4,
        validation_data=db_test,
        validation_freq=1
    )

    network.evaluate(db_test)

    network.save_weights('ckpt/weights.ckpt')   # 只保存参数
    del network
    print('save to ckpt/weights.ckpt !')


    # load and test
    network = MyModel()
    network.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    network.load_weights('ckpt/weights.ckpt')
    network.evaluate(db_test)





