import tensorflow as tf
from zookeeper import ComponentField, Field, cli, task

from larq_zoo.literature.real_to_bin_nets import (
    RealToBinNetBANFactory,
    RealToBinNetBNNFactory,
    RealToBinNetFPFactory,
    ResNet18FPFactory,
    StrongBaselineNetBANFactory,
    StrongBaselineNetBNNFactory,
)
from larq_zoo.literature.reactnet import (
    ResNet34Factory,
    ReActNetBANFactory,
    ReActNetBNNFactory
)
from larq_zoo.training.datasets import ImageNet
from larq_zoo.training.knowledge_distillation.multi_stage_training import (
    LarqZooModelTrainingPhase,
    MultiStageExperiment,
)
from larq_zoo.training.learning_schedules import CosineDecayWithWarmup, R2BStepSchedule
from zookeeper.tf import experiment


# --------- Real-to-Binary: Strong Baseline Model training -------------

# Note: we provide the below classes primarily as an example of how to use Zoo's multi-
# stage infrastructure to implement a set of training steps - this is not the exact
# setup we used to train the pretrained weights available for our implementation of
# Real-to- Binary nets. Therefore, these exact steps are not expected to reproduce our
# weights or reported accuracies.
#
# See related discussions here:
# - https://github.com/larq/zoo/issues/233
# - https://github.com/larq/zoo/issues/196


@task
class TrainR2BStrongBaselineBAN(LarqZooModelTrainingPhase):
    stage = Field(0)

    dataset = ComponentField(ImageNet)

    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.1)
    epochs: int = Field(75)
    batch_size: int = Field(8)
    # amount_of_images: int = Field(1281167)
    warmup_duration: int = Field(5)

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            R2BStepSchedule(
                initial_learning_rate=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch,
                decay_fraction=self.learning_rate_decay,
            )
        )
    )

    student_model = ComponentField(StrongBaselineNetBANFactory)


@task
class TrainR2BStrongBaselineBNN(TrainR2BStrongBaselineBAN):
    stage = Field(1)
    learning_rate: float = Field(2e-4)
    student_model = ComponentField(StrongBaselineNetBNNFactory)
    initialize_student_weights_from = Field("baseline_ban")


@task
class TrainR2BStrongBaseline(MultiStageExperiment):
    stage_0 = ComponentField(TrainR2BStrongBaselineBAN)
    stage_1 = ComponentField(TrainR2BStrongBaselineBNN)


# --------- Real-to-Binary: Full Model training -------------


@task
class TrainFPResnet18(LarqZooModelTrainingPhase):
    stage = Field(0)
    dataset = ComponentField(ImageNet)
    # learning_rate: float = Field(1e-1)
    learning_rate: float = Field(1e-3)
    epochs: int = Field(100)
    batch_size: int = Field(512)
    # amount_of_images: int = Field(1281167)
    warmup_duration: int = Field(5)
    experiment_name: str = Field("FPresnet18")

    # optimizer = Field(
    #     lambda self: tf.keras.optimizers.SGD(
    #         CosineDecayWithWarmup(
    #             max_learning_rate=self.learning_rate,
    #             warmup_steps=self.warmup_duration * self.steps_per_epoch,
    #             decay_steps=self.epochs * self.steps_per_epoch,
    #         )
    #     )
    # )

    optimizer = Field(lambda self: tf.keras.optimizers.Adam(
        CosineDecayWithWarmup(
            max_learning_rate=self.learning_rate,
            warmup_steps=self.warmup_duration * self.steps_per_epoch,
            decay_steps=(self.epochs - self.warmup_duration) * self.steps_per_epoch,
        )
    ))

    student_model = ComponentField(ResNet18FPFactory)


@task
class TrainR2BBFP(TrainFPResnet18):
    stage = Field(1)
    learning_rate: float = Field(1e-3)
    learning_rate_decay: float = Field(0.3)
    epochs: int = Field(75)
    batch_size: int = Field(256)
    experiment_name: str = Field("R2BBFP")

    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            R2BStepSchedule(
                initial_learning_rate=self.learning_rate,
                steps_per_epoch=self.steps_per_epoch,
                decay_fraction=self.learning_rate_decay,
            )
        )
    )

    teacher_model = ComponentField(ResNet18FPFactory)
    initialize_teacher_weights_from = Field("resnet_fp")
    student_model = ComponentField(RealToBinNetFPFactory)

    classification_weight = Field(1.0)
    attention_matching_weight = Field(30.0)
    output_matching_weight = Field(3.0)

    attention_matching_volume_names = Field(
        lambda: [f"block_{b}_out" for b in range(2, 10)]
    )


@task
class TrainR2BBAN(TrainR2BBFP):
    stage = Field(2)
    learning_rate: float = Field(1e-3)

    teacher_model = ComponentField(RealToBinNetFPFactory)
    student_model = ComponentField(RealToBinNetBANFactory)

    initialize_teacher_weights_from = Field("r2b_fp")
    experiment_name: str = Field("R2BBAN")


@task
class TrainR2BBNN(TrainR2BBFP):
    stage = Field(3)
    learning_rate: float = Field(2e-4)

    classification_weight = Field(1.0)
    attention_matching_weight = Field(0.0)
    output_matching_weight = Field(0.8)
    output_matching_softmax_temperature = Field(1.0)

    teacher_model = ComponentField(RealToBinNetBANFactory)
    student_model = ComponentField(RealToBinNetBNNFactory)

    initialize_teacher_weights_from = Field("r2b_ban")
    initialize_student_weights_from = Field("r2b_ban")
    experiment_name: str = Field("R2BBNN")


@task
class TrainR2BBNNAlternative(TrainR2BBNN):
    """We deviate slightly from Martinez et. al. here"""

    warmup_duration = Field(10)
    optimizer = Field(
        lambda self: tf.keras.optimizers.Adam(
            CosineDecayWithWarmup(
                max_learning_rate=self.learning_rate,
                warmup_steps=self.steps_per_epoch * self.warmup_duration,
                decay_steps=self.steps_per_epoch * self.epochs,
            )
        )
    )


@task
class TrainR2B(MultiStageExperiment):
    stage_0 = ComponentField(TrainFPResnet18)
    stage_1 = ComponentField(TrainR2BBFP)
    stage_2 = ComponentField(TrainR2BBAN)
    stage_3 = ComponentField(TrainR2BBNNAlternative)



# --------- ReActNet: Full Model training -------------


@task
class TrainReActNetBAN(LarqZooModelTrainingPhase):
    use_progress_bar = Field(True)
    stage = Field(0)
    dataset = ComponentField(ImageNet)

    learning_rate: float = Field(5e-4)
    epochs: int = Field(120)
    batch_size: int = Field(256)

    decay_schedule: str = Field("linear")
    allow_missing_teacher_weights = Field(True)
    # teacher_model = ComponentField(ResNet34Factory)
    student_model = ComponentField(ReActNetBANFactory)
    classification_weight = Field(1.0)
    # output_matching_weight = Field(1e-4)
    # output_matching_softmax_temperature = Field(1.0)
    experiment_name: str = Field("ReActNet_C_dw_approx_sign_BAN")

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
class TrainReActNetBNN(TrainReActNetBAN):
    stage = Field(1)
    # teacher_model = ComponentField(ResNet34Factory)
    student_model = ComponentField(ReActNetBNNFactory)
    initialize_student_weights_from = Field("reactnet_ban")
    experiment_name: str = Field("ReActNet_C_dw_approx_sign_BNN")


@task
class TrainReActNet(MultiStageExperiment):
    stage_0 = ComponentField(TrainReActNetBAN)
    stage_1 = ComponentField(TrainReActNetBNN)


if __name__ == "__main__":
    cli()
