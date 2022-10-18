import larq as lq
from typing import Sequence

import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.literature.binary_alex_net import BinaryAlexNetFactory
from larq_zoo.literature.birealnet import BiRealNetFactory
from larq_zoo.literature.reactnet import ReActNetBNNFactory
from larq_zoo.literature.densenet import (
    BinaryDenseNet,
    BinaryDenseNet28Factory,
    BinaryDenseNet37DilatedFactory,
    BinaryDenseNet37Factory,
    BinaryDenseNet45Factory,
)
from larq_zoo.literature.dorefanet import DoReFaNetFactory
from larq_zoo.literature.resnet_e import BinaryResNetE18Factory
from larq_zoo.literature.xnornet import XNORNetFactory
from larq_zoo.literature.lab_bnn import LabBNNFactory
from larq_zoo.training.train import TrainLarqZooModel


@task
class TrainBinaryAlexNet(TrainLarqZooModel):
    model = ComponentField(BinaryAlexNetFactory)

    batch_size: int = Field(512)
    epochs: int = Field(150)

    def learning_rate_schedule(self, epoch):
        return 1e-2 * 0.5 ** (epoch // 10)

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(self.learning_rate_schedule(0))
    )


@task
class TrainBiRealNet(TrainLarqZooModel):
    model = ComponentField(BiRealNetFactory)

    epochs = Field(150)
    batch_size = Field(256)
    lab_blocks = Field((True, True, True, True))
    resume_from = Field(None)

    learning_rate: float = Field(2.5e-3) 
    # learning_rate: float = Field(1e-5)   
    decay_schedule: str = Field("linear")

    @Field
    def optimizer(self):
        if self.decay_schedule == "linear_cosine":
            lr = tf.keras.experimental.LinearCosineDecay(self.learning_rate, (1281167/self.batch_size))
        elif self.decay_schedule == "linear":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                self.learning_rate, (1281167/self.batch_size)*(self.epochs), end_learning_rate=1e-6, power=1.0
            )
        else:
            lr = self.learning_rate
        return tf.keras.optimizers.Adam(lr)
@task
class TrainLabBNN(TrainLarqZooModel):
    model = ComponentField(LabBNNFactory)

    epochs = Field(100)
    batch_size = Field(256)
    lab_blocks = Field((True, True, True, True))
    resume_from = Field(None)

    learning_rate: float = Field(2.5e-3)    
    decay_schedule: str = Field("linear")

    @Field
    def optimizer(self):
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.learning_rate, (1281167/self.batch_size)*(self.epochs), end_learning_rate=1e-6, power=1.0
        )
        return tf.keras.optimizers.Adam(lr)
@task
class TrainReActNetFromScratch(TrainLarqZooModel):
    use_progress_bar = Field(True)

    learning_rate: float = Field(2.5e-4)
    # learning_rate: float = Field(1e-5)
    epochs: int = Field(75)
    batch_size: int = Field(128)

    decay_schedule: str = Field("linear")
    model = ComponentField(ReActNetBNNFactory)
    # resume_from = Field(None)
    # resume_from = Field("/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/sfalkena/larq/zookeeper-logs/ImageNet/TrainReActNetBasic/ran_org_0000/20220209_1437")
    resume_from = Field("/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/sfalkena/larq/zookeeper-logs/ImageNet/TrainReActNetBasic/reactnet_dw_mult_bin_1111/20220209_1437")
    
    @Field
    def optimizer(self):
        if self.decay_schedule == "linear":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                self.learning_rate, 750684, end_learning_rate=0, power=1.0
            )
        else:
            lr = self.learning_rate
        return tf.keras.optimizers.Adam(lr)





@task
class TrainBinaryResNetE18(TrainLarqZooModel):
    model = ComponentField(BinaryResNetE18Factory)

    epochs = Field(120)
    batch_size = Field(1024)

    learning_rate: float = Field(0.004)
    learning_factor: float = Field(0.3)
    learning_steps: Sequence[int] = Field((70, 90, 110))

    def learning_rate_schedule(self, epoch):
        lr = self.learning_rate
        for step in self.learning_steps:
            if epoch < step:
                return lr
            lr *= self.learning_factor
        return lr

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-8)
    )


@task
class TrainBinaryDenseNet28(TrainLarqZooModel):
    model: BinaryDenseNet = ComponentField(BinaryDenseNet28Factory)

    epochs = Field(120)
    batch_size = Field(256)

    learning_rate: float = Field(4e-3)
    learning_factor: float = Field(0.1)
    learning_steps: Sequence[int] = Field((100, 110))

    def learning_rate_schedule(self, epoch):
        lr = self.learning_rate
        for step in self.learning_steps:
            if epoch < step:
                return lr
            lr *= self.learning_factor
        return lr

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-8)
    )


@task
class TrainBinaryDenseNet37(TrainBinaryDenseNet28):
    model = ComponentField(BinaryDenseNet37Factory)
    batch_size = Field(192)


@task
class TrainBinaryDenseNet37Dilated(TrainBinaryDenseNet37):
    model = ComponentField(BinaryDenseNet37DilatedFactory)
    epochs = Field(80)
    batch_size = Field(256)
    learning_steps = Field((60, 70))


@task
class TrainBinaryDenseNet45(TrainBinaryDenseNet28):
    model = ComponentField(BinaryDenseNet45Factory)
    epochs = Field(125)
    batch_size = Field(384)
    learning_rate = Field(0.008)
    learning_steps = Field((80, 100))


@task
class TrainXNORNet(TrainLarqZooModel):
    model = ComponentField(XNORNetFactory)

    epochs = Field(60)
    batch_size = Field(128)    
    lab_blocks = Field((True, True)) # Ugly fix for now
    resume_from = Field(None)
    initial_lr: float = Field(1e-4)

    def learning_rate_schedule(self, epoch):
        epoch_dec_1 = 19
        epoch_dec_2 = 30
        epoch_dec_3 = 44
        epoch_dec_4 = 53
        epoch_dec_5 = 66
        epoch_dec_6 = 76
        epoch_dec_7 = 86
        if epoch < epoch_dec_1:
            return self.initial_lr
        elif epoch < epoch_dec_2:
            return self.initial_lr * 0.5
        elif epoch < epoch_dec_3:
            return self.initial_lr * 0.1
        elif epoch < epoch_dec_4:
            return self.initial_lr * 0.1 * 0.5
        elif epoch < epoch_dec_5:
            return self.initial_lr * 0.01
        elif epoch < epoch_dec_6:
            return self.initial_lr * 0.01 * 0.5
        elif epoch < epoch_dec_7:
            return self.initial_lr * 0.01 * 0.1
        else:
            return self.initial_lr * 0.001 * 0.1

    optimizer = Field(lambda self: tf.keras.optimizers.Adam(self.initial_lr))


@task
class TrainDoReFaNet(TrainLarqZooModel):
    model = ComponentField(DoReFaNetFactory)

    epochs = Field(90)
    batch_size = Field(256)

    learning_rate: float = Field(2e-4)
    decay_start: int = Field(60)
    decay_step_2: int = Field(75)
    fast_decay_start: int = Field(82)

    def learning_rate_schedule(self, epoch):
        if epoch < self.decay_start:
            return self.learning_rate
        elif epoch < self.decay_step_2:
            return self.learning_rate * 0.2
        elif epoch < self.fast_decay_start:
            return self.learning_rate * 0.2 * 0.2
        else:
            return (
                self.learning_rate
                * 0.2
                * 0.2
                * 0.1 ** ((epoch - self.fast_decay_start) // 2 + 1)
            )

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)
    )


if __name__ == "__main__":
    cli()
