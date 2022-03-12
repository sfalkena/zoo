import functools
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Union

import click
import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import Field
from zookeeper.tf import Experiment
from larq_zoo.core import utils
import gc

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, checkpoint_dir, model, optimizer):

        super(EpochModelCheckpoint, self).__init__(filepath=checkpoint_dir)

        self.ckpt = tf.train.Checkpoint(completed_epochs=tf.Variable(0,trainable=False,dtype='int32'), optimizer=optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=1)

    def on_epoch_end(self,epoch,logs=None):        
        self.ckpt.completed_epochs.assign(epoch)
        self.manager.save(checkpoint_number=epoch)
        print( f"Epoch checkpoint {self.ckpt.completed_epochs.numpy()}  saved to: {self.manager.latest_checkpoint}" ) 
        print(logs)
               
class TrainLarqZooModel(Experiment):
    # Save model checkpoints.
    use_model_checkpointing: bool = Field(True)

    # Log metrics to Tensorboard.
    use_tensorboard: bool = Field(True)

    # Use a per-batch progress bar (as opposed to per-epoch).
    use_progress_bar: bool = Field(False)

    # How often to run validation.
    validation_frequency: int = Field(1)

    # Whether or not to save models at the end.
    save_weights: bool = Field(True)

    # Where to store output.
    @Field
    def output_dir(self) -> Union[str, os.PathLike]:
        if self.resume_from:
            print(f"resuming path: {self.resume_from}")
            return Path(self.resume_from)
        else:
            foldername = self.experiment_name+'_'+''.join([str(int(block == True)) for block in self.convbin_blocks])
            return (
                Path(PATH_OF_REPOSITORY)
                / "zookeeper-logs"
                / self.dataset.__class__.__name__
                / self.__class__.__name__
                / foldername
                / datetime.now().strftime("%Y%m%d_%H%M")
            )

    @property
    def checkpoint_dir(self):
        return Path(self.output_dir) / "tf_ckpts/"

    metrics: List[Union[Callable[[tf.Tensor, tf.Tensor], float], str]] = Field(
        lambda: ["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
    )

    loss = Field("sparse_categorical_crossentropy")

    @property
    def steps_per_epoch(self):
        return self.dataset.num_examples("train") // self.batch_size

    @Field
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        if self.use_model_checkpointing:
            callbacks.append(
                EpochModelCheckpoint(self.checkpoint_dir, self.model, self.optimizer)
            )
        if hasattr(self, "learning_rate_schedule"):
            callbacks.append(
                keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
            )
        if self.use_tensorboard:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.output_dir, write_graph=False, histogram_freq=1, profile_batch=0
                )
            )
        callbacks.append(MyCustomCallback())
        return callbacks

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


        train_data, num_train_examples = self.dataset.train(
            decoders=self.preprocessing.decoders
        )
        if self.dataset.__class__.__name__ == 'ImageNet':
            train_cache = PATH_TO_STORE_IMAGENET_TRAIN_CACHE_FILE 
        train_data = (
            train_data.cache(train_cache)
            .shuffle(10 * self.batch_size)
            .repeat()
            .map(
                functools.partial(self.preprocessing, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(self.batch_size)
            .prefetch(1)
        )

        validation_data, num_validation_examples = self.dataset.validation(
            decoders=self.preprocessing.decoders
        )
        if self.dataset.__class__.__name__ == 'ImageNet':
            val_cache = PATH_TO_STORE_IMAGENET_VAL_CACHE_FILE
        validation_data = (
            validation_data.cache(val_cache)
            .repeat()
            .map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(1)
        )
        with utils.get_distribution_scope(self.batch_size):


            ckpt = tf.train.Checkpoint(completed_epochs=tf.Variable(0,trainable=False,dtype='int32'), model=self.model)
            manager = tf.train.CheckpointManager(ckpt, self.checkpoint_dir, max_to_keep=2, keep_checkpoint_every_n_hours=10)

            # Restore last Epoch
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print(f"Restored epoch ckpt from {manager.latest_checkpoint}, value is ",ckpt.completed_epochs.numpy())
            else:
                print("Initializing from scratch.")
            
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics,
            )
            lq.models.summary(self.model)

        completed_epochs=ckpt.completed_epochs.numpy()

        click.secho(str(self))

        # Print FLOPS
        # forward_pass = tf.function(
        #     self.model.call,
        #     input_signature=[tf.TensorSpec(shape=(1,) + self.model.input_shape[1:])])

        # graph_info = profile(forward_pass.get_concrete_function().graph,
        #                         options=ProfileOptionBuilder.float_operation())
        # flops = graph_info.total_float_ops // 2
        # print('Flops: {:,}'.format(flops))

        self.model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=math.ceil(num_train_examples / self.batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(num_validation_examples / self.batch_size),
            validation_freq=self.validation_frequency,
            verbose=1 if self.use_progress_bar else 2,
            initial_epoch=completed_epochs,
            callbacks=self.callbacks,
        )
        gc.collect()

        # Save model, weights, and config JSON.
        if self.save_weights:
            self.model.save(str(Path(self.output_dir) / f"{self.model.name}.h5"))
            self.model.save_weights(
                str(Path(self.output_dir) / f"{self.model.name}_weights.h5")
            )
            with open(
                Path(self.output_dir) / f"{self.model.name}.json", "w"
            ) as json_file:
                json_file.write(self.model.to_json())
