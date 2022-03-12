import larq as lq
import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.sota.quicknet import (
    QuickNetFactory,
    QuickNetLargeFactory,
    QuickNetSmallFactory,
)
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup
from larq_zoo.training.train import TrainLarqZooModel


@task
class TrainQuickNet(TrainLarqZooModel):
    model = ComponentField(QuickNetFactory)
    epochs = Field(150)
    batch_size = Field(512)
    lab_blocks = Field((True, True, True, True))
    model.lab_blocks = lab_blocks
    resume_from = Field(None)

    @Field
    def optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=CosineDecayWithWarmup(
                max_learning_rate=2.5e-3,
                warmup_steps=self.steps_per_epoch * 5,
                decay_steps=self.steps_per_epoch * self.epochs,
            )
        )

@task
class TrainQuickNetSmall(TrainQuickNet):
    model = ComponentField(QuickNetSmallFactory)


@task
class TrainQuickNetLarge(TrainQuickNet):
    model = ComponentField(QuickNetLargeFactory)


if __name__ == "__main__":
    cli()