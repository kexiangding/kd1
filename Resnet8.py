
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Acativation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:  # 维数不为1的时候需要进行下采样
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
            self.downsample.add(layers.BatchNormalization())

        else:
            self.downsample = lambda x: x

    # self.stride = stride

    def call(self, inputs, training=None):
        residual = self.downsample(inputs)  # 原来的x

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = layers.add([bn2, residual])  # shortcut
        out = self.relu(add)  # 等价于out = tf.nn.relu(add)
        return out


class ResNet(keras.Model):

    def __int__(self, layer_dims, num_classes=100):  # [2,2,2,2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w]
        self.avgpool = layers.GlobalAveragePooling2D()  # 功能层
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, trianing=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b,c]
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        # may downsample
        res_blocks = keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])


import os
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import resnet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.random.set_seed(2345)
#from resnet import resnet18


def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 256
# [32, 32, 3], [10k, 1]
(x, y), (x_val, y_val) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_val = tf.squeeze(y_val, axis=1)  # 注意维度变换
print(x.shape, y.shape, x_val.shape, y_val.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('batch: ', sample[0].shape, sample[1].shape)


def main():
    model = resnet18()
    model.summary()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = optimizers.Adam(lr=1e-4)

    # 拼接需要训练的参数 [1,2] + [3,4] = [1,2,3,4]
    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b,32,32,3] => [b,1,1,512]
                logits = model(x)

                y_onehot = tf.one_hot(y, depth=100)  # [50k, 10]
                # y_val_onehot = tf.one_hot(y_val, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trianabel_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss: ', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print('acc: ', acc)


#if __name__ == '__main__':
#    main()
main()