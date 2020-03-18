import tensorflow as tf
from tensorflow.keras import datasets

#tf.enable_eager_execution()
# (x, y), (test_x, test_y) = datasets.mnist.load_data()
(x, y), (test_x, test_y) = datasets.fashion_mnist.load_data()
print('datasets:', x.shape, y.shape, test_x.shape, test_y.shape)

test_x = tf.convert_to_tensor(test_x, dtype=tf.float32) / 255
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255

lr = 0.001
db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(128)

db_iter = iter(db)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)

w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 125], stddev=0.1))
b2 = tf.Variable(tf.zeros([125]))
w3 = tf.Variable(tf.random.truncated_normal([125, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

acc = 0
while acc < 0.95:
    for step, (x, y) in enumerate(db):
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            y_onehot = tf.one_hot(y, depth=10)

            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print("step", step / 100, "loss:", loss)

    correct_sum, total = 0, 0
    for step, (test_x, test_y) in enumerate(test_db):
        test_x = tf.reshape(test_x, [-1, 28 * 28])
        h1 = test_x @ w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        out = h2 @ w3 + b3

        res = tf.nn.softmax(out, axis=1)
        res = tf.argmax(res, axis=1)
        res = tf.cast(res, dtype=tf.int32)
        y = tf.cast(test_y, dtype=tf.int32)
        correct = tf.cast(tf.equal(res, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        correct_sum += correct
        total += test_x.shape[0]

    acc = correct_sum / total
    acc = float(acc)
    print("accuracy:", acc)

print("w1:", w1, "w2:", w2, "w3:", w3)
print("b1:", b1, "b2:", b2, "b3:", b3)


