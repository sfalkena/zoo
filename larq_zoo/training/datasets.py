from zookeeper import Field, component
from zookeeper.tf import TFDSDataset


@component
class ImageNet(TFDSDataset):
    name = Field("imagenet2012:5.1.*")
    train_split = Field("train")
    validation_split = Field("validation")
    data_dir = Field("/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/sfalkena/datasets/imagenet/")
    download = Field(True)


@component
class Cifar10(TFDSDataset):
    name = Field("cifar10:3.0.*")
    train_split = Field("train")
    validation_split = Field("test")


@component
class Mnist(TFDSDataset):
    name = Field("mnist:3.0.*")
    train_split = Field("train")
    validation_split = Field("test")


@component
class OxfordFlowers(TFDSDataset):
    name = Field("oxford_flowers102:2.0.*")
    train_split = Field("train+validation")
    validation_split = Field("test")
