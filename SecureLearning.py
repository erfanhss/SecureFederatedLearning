import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import SecAgg

@tf.function
def calculateGradients(xTrain, model, yTrain):
    with tf.GradientTape() as tape:
        logits = model(xTrain, training=True)
        lossValue = loss_fn(yTrain, logits)
    grads = tape.gradient(lossValue, model.trainable_weights)
    return grads, lossValue


def random_matrix(p, s, w, seed):
  counts = tf.cast(tf.random.stateless_normal([1], seed, mean= s*w*p*2, stddev=np.sqrt(s*w*2*p*(1-2*p))), dtype=tf.int32)
  rows = tf.random.stateless_uniform(counts, seed, minval=0, maxval=s, dtype=tf.int64)
  cols = tf.random.stateless_uniform(counts, seed, minval=0, maxval=w, dtype=tf.int64)
  vals = tf.cast(tf.random.stateless_binomial(counts, seed, 1, probs=0.5)*2-1, tf.float32)
  return tf.sparse.reorder(tf.SparseTensor(tf.stack([rows, cols], axis=-1), vals, [s, w]))


inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
numWorkers = 8
batchSizePerWorker = 128
numEpochs = 100*60
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()
xTrain = np.reshape(xTrain, (-1, 784))
xTest = np.reshape(xTest, (-1, 784))
randomIdx = np.random.permutation(len(xTrain))
workerDataIdx = np.array_split(randomIdx, numWorkers)
workerDataSetX = [xTrain[idx] for idx in workerDataIdx]
workerDataSetY = [yTrain[idx] for idx in workerDataIdx]

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
  name='test_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
for epoch in range(numEpochs):
    train_loss = tf.keras.metrics.Mean()
    workerGrads = []
    phi = random_matrix(alpha / 2 / samples, samples, params_count, seed=epoch)
    for worker in range(numWorkers):
        batchIdx = np.random.permutation(len(workerDataSetX[worker]))[0:batchSizePerWorker]
        batchX = workerDataSetX[worker][batchIdx]
        batchY = workerDataSetY[worker][batchIdx]
        gradients, loss = calculateGradients(batchX, model, batchY)
        train_loss.update_state(loss)
        flat_grad = []
        shapes = []
        for arr in gradients:
            flat_grad.append(tf.reshape(arr, [-1, 1]))
            shapes.append(tf.shape(arr))
        flat_grad = tf.concat(flat_grad, axis=0)
        error_compensated = lr * flat_grad + error[worker]
        compressed = tf.sign(tf.sparse.sparse_dense_matmul(phi, error_compensated)) + 2
        workerGrads.append(np.ndarray.astype(compressed.numpy(), dtype=np.int64))
        beta = tf.norm(error_compensated, ord=1) / params_count / (1 + alpha) / (1 + alpha * r)
        recov = beta * tf.sparse.sparse_dense_matmul(tf.sparse.transpose(phi), compressed)
        error[worker] = error_compensated - recov
    secureAgggregationResult = SecAgg.secureAggregtion(workerGrads, 13, numWorkers // 2)
    aggregatedGradients = secureAgggregationResult[0] - 2*numWorkers
    output = []
    cntr = 0
    for shape in shapes:
        num_elements = tf.math.reduce_prod(shape)
        params = tf.reshape(aggregatedGradients[cntr:cntr + num_elements], shape)
        params = tf.cast(params, tf.float32)
        cntr += num_elements
        output.append(params)
    optimizer.apply_gradients(zip(output, model.trainable_weights))

    if epoch+1 % 60 == 0:
        testIdx = np.random.permutation(len(xTest))
        testBatchIdx = np.array_split(testIdx, 60)
        for batchIdx in testBatchIdx:
            logits = model(xTest[batchIdx], training=False)
            lossValue = loss_fn(yTest[batchIdx], logits)
            test_accuracy.update_state(yTest[batchIdx], logits)
            test_loss.update_state(lossValue)
        print('Epoch: ' + (epoch+1)/60)
        print('Test Loss: ' + test_loss.result().numpy())
        print('Test Accuracy: ' + test_accuracy.result().numpy())
