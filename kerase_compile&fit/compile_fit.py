import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

batchsz = 128
x = tf.reshape(x, [-1, 28 * 28])
y = tf.one_hot(y, depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

x_test = tf.reshape(x_test, [-1, 28 * 28])
y_test = tf.one_hot(y_test, depth=10)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)



def main():
    network = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu)
    ])
    network.build(input_shape=[None, 28*28])
    network.summary()

    network.compile(
            optimizer=optimizers.Adam(lr=0.01),    # 指定优化器
            loss=tf.losses.CategoricalCrossentropy(from_logits=True),   # 指定loss函数
            metrics=['accuracy']     # 指定测试标准
        )

    network.fit(
            db,   # 要训练的数据集
            epochs=10,    # 训练的周期
            validation_data=db_test,    # 用于做测试的数据集,一般写作ds_val
            validation_freq=2
        )

    network.evaluate(db_test)    # 训练完后对模型的评估,传入一个数据集

    pred = network(x)


if __name__ == "__main__":
    main()
