import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, Sequential, optimizers
from resnet import resnet18


def preproccess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (test_x, test_y) = datasets.cifar10.load_data()

batchsz = 128
y = tf.squeeze(y, axis=1)
train_set = tf.data.Dataset.from_tensor_slices((x, y))
train_set = train_set.map(preproccess).shuffle(1000).batch(batchsz)

test_y = tf.squeeze(test_y, axis=1)
test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_set = test_set.map(preproccess).batch(batchsz)

db_iter = iter(test_set)
sample = next(db_iter)
print('batch:', sample[0].shape, sample[1].shape)


def main():
    model = resnet18()

    model.build(input_shape=[None, 32, 32, 3])
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(30):
        for step, (x, y) in enumerate(train_set):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variable)
            optimizer.apply_gradients(zip(grads, model.trainable_variable))

            if step % 100 == 0:
                print(epoch, 'step:', step/100, ',loss:', float(loss))

        print('=====================\n')
        
        total_num = 0
        total_correct = 0
        for x, y in test_set:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
        

if __name__ == '__main__':
    main()
    #print(x.shape, y.shape)
