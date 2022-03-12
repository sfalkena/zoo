from typing import Optional, Sequence, Union

import larq as lq
import tensorflow as tf
from zookeeper import Field, factory

from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory


@factory
class LabBNNFactory(ModelFactory):
    """Implementation of [LAB-BNN]"""
    
    filters: int = Field(64)
    input_quantizer = lq.quantizers.LAB()
    kernel_quantizer = "magnitude_aware_sign"
    kernel_constraint = "weight_clip"
    convbin_blocks: Sequence[bool] = Field()

    kernel_initializer: Union[tf.keras.initializers.Initializer, str] = Field(
        "glorot_normal"
    )

    def residual_block(
        self, x, use_binconv: bool, double_filters: bool = False, filters: Optional[int] = None
    ) -> tf.Tensor:
        assert not (double_filters and filters)

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
            # input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        return x

    def build(self) -> tf.keras.models.Model:
        
        x = lq.layers.QuantConv2D(
            self.filters // 4,
            (3, 3),
            kernel_initializer="he_normal",
            padding="same",
            strides=2,
            use_bias=False,
        )(self.image_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = lq.layers.QuantDepthwiseConv2D(
            (3, 3),
            padding="same",
            strides=2,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, center=False)(x)

        x = lq.layers.QuantConv2D(
            self.filters,
            1,
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
        out = tf.keras.layers.BatchNormalization()(x)
    
        # Layer 2
        out = self.residual_block(out, self.convbin_blocks[0], filters=self.filters)

        # Layer 3 - 5
        for _ in range(3):
            out = self.residual_block(out, self.convbin_blocks[0])

        # Layer 6 - 17
        for i in range(3):
            out = self.residual_block(out, self.convbin_blocks[i+1], double_filters=True)
            for _ in range(3):
                out = self.residual_block(out, self.convbin_blocks[i+1])


        # Layer 18
        if self.include_top:
            out = utils.global_pool(out)
            out = tf.keras.layers.Dense(self.num_classes)(out)
            out = tf.keras.layers.Activation("softmax", dtype="float32")(out)

        model = tf.keras.Model(inputs=self.image_input, outputs=out, name="birealnet18")

        return model


def LabBNN(
    *,  # Keyword arguments only
    input_shape: Optional[Sequence[Optional[int]]] = None,
    input_tensor: Optional[utils.TensorType] = None,
    include_top: bool = True,
    num_classes: int = 1000,
) -> tf.keras.models.Model:
    """Instantiates the LAB-BNN architecture.

    # ImageNet Metrics

    | Top-1 Accuracy | Top-5 Accuracy | Memory  |
    | -------------- | -------------- | ------- |
    | 62.4%          | 85.5%          | 4.62 MB |

    # Arguments
        input_shape: Optional shape tuple, to be specified if you would like to use a
            model with an input image resolution that is not (224, 224, 3).
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape.

    # References
        - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
            Representational Capability and Advanced Training
            Algorithm](https://arxiv.org/abs/1808.00278)
    """
    return LabBNNFactory(
        include_top=include_top,
        input_tensor=input_tensor,
        input_shape=input_shape,
        num_classes=num_classes,
    ).build()