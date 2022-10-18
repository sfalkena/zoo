from typing import Optional, Sequence, Union

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


@factory
class BiRealNetFactory(ModelFactory):
    """Implementation of [Bi-Real Net](https://arxiv.org/abs/1808.00278)"""

    filters: int = Field(64)

    kernel_quantizer = "magnitude_aware_sign"
    kernel_constraint = "weight_clip"
    lab_blocks: Sequence[bool] = Field()

    kernel_initializer: Union[tf.keras.initializers.Initializer, str] = Field(
        "glorot_normal"
    )

    def residual_block(
        self, x, use_lab: bool, double_filters: bool = False, filters: Optional[int] = None
    ) -> tf.Tensor:
        assert not (double_filters and filters)

        self.input_quantizer = lq.quantizers.Sauvola() if use_lab else lq.quantizers.SteSign()
        # Compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x

        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters,
                (1, 1),
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            (3, 3),
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)

        return tf.keras.layers.add([x, shortcut])

    def build(self) -> tf.keras.models.Model:
        # Layer 1
        out = tf.keras.layers.Conv2D(
            self.filters,
            (7, 7),
            strides=2,
            kernel_initializer=self.kernel_initializer,
            padding="same",
            use_bias=False,
        )(self.image_input)
        out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
        out = tf.keras.layers.MaxPool2D((3, 3), strides=2, padding="same")(out)

        # Layer 2
        out = self.residual_block(out, self.lab_blocks[0], filters=self.filters)

        # Layer 3 - 5
        for _ in range(3):
            out = self.residual_block(out, self.lab_blocks[0])

        # Layer 6 - 17
        for i in range(3):
            out = self.residual_block(out, self.lab_blocks[i+1], double_filters=True)
            for _ in range(3):
                out = self.residual_block(out, self.lab_blocks[i+1])

        # Layer 18
        if self.include_top:
            out = utils.global_pool(out)
            out = tf.keras.layers.Dense(self.num_classes)(out)
            out = tf.keras.layers.Activation("softmax", dtype="float32")(out)

        model = tf.keras.Model(inputs=self.image_input, outputs=out, name="birealnet18")

        return model


def BiRealNet(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    include_top: bool = True,
    num_classes: int = 1000,
    lab_blocks: Sequence[int],
) -> tf.keras.models.Model:
    """Instantiates the Bi-Real Net architecture.

    Optionally loads weights pre-trained on ImageNet.

    ```netron
    birealnet-v0.3.0/birealnet.json
    ```
    ```summary
    literature.BiRealNet
    ```
    ```plot-altair
    /plots/birealnet.vg.json
    ```

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Parameters | Memory  |
    | -------------- | -------------- | ---------- | ------- |
    | 57.47 %        | 79.84 %        | 11 699 112 | 4.03 MB |

    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        weights: one of `None` (random initialization), "imagenet" (pre-training on
            ImageNet), or the path to the weights file to be loaded.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        lab_blocks: in which of the blocks to apply Lab binarization given in a list
            of four booleans.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.

    # References
        - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
            Representational Capability and Advanced Training
            Algorithm](https://arxiv.org/abs/1808.00278)
    """
    return BiRealNetFactory(
        include_top=include_top,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
        lab_blocks=lab_blocks,
    ).build()
