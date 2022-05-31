import os
import glob
from tqdm import tqdm
import cv2
import albumentations as A

import tensorflow as tf

class TFRWriter:
    def __init__(self):
        self.main_dir = "sRGB2XYZ"
        self.save_path = os.path.join(self.main_dir, "shards")
        self.resize_shape = (256, 256)
        self.transform = A.Compose(
            [
                A.Flip(),
                A.RandomCrop(256, 256)
            ],
            additional_targets={'XYZ_image': 'image'}
        )

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def serialize_example(self, sample):
        feature = {
            "XYZ_image": self._bytes_feature(sample[0]),
            "sRGB_image": self._bytes_feature(sample[1]),
            "filename": self._bytes_feature(sample[2])}

        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    def get_image(self, sample_path):
        image = cv2.imread(sample_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize_shape)
        return image

    def get_samples(self, train_set):
        path = f"{self.main_dir}/sRGB_{train_set}/*.JPG"
        return glob.glob(path)

    def get_sample_data(self, sample):
        xyz_filename = sample.replace("sRGB_", "XYZ_").replace(".JPG", ".png")
        assert os.path.exists(xyz_filename), f"{xyz_filename} file does not exist."

        srgb_image = self.get_image(sample)
        xyz_image = self.get_image(xyz_filename)
        transformed = self.transform(image=srgb_image, XYZ_image=xyz_image)
        filename = os.path.basename(sample)
                
        return [
            tf.io.serialize_tensor(transformed['XYZ_image'], name="XYZ_image"),
            tf.io.serialize_tensor(transformed['image'], name="sRGB_image"),
            tf.io.serialize_tensor(filename, name="filename")]

    def write(self):
        for train_set in ["training", "validation", "testing"]:
            shard_path = os.path.join(self.save_path, f"{train_set}.tfrec")
            samples = self.get_samples(train_set)
            with tf.io.TFRecordWriter(shard_path) as f:
                for sample in tqdm(samples, total=len(samples), desc=f"{train_set}"):
                    sample_data = self.get_sample_data(sample)
                    f.write(self.serialize_example(sample_data))

class DataLoader:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.buffer_size = 64

    def read_tfrecord(self, example):
        feature_description = {
            'XYZ_image': tf.io.FixedLenFeature([], tf.string),
            'sRGB_image': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string)}
        
        example = tf.io.parse_single_example(example, feature_description)

        for key in example:
            if key != 'filename':
                example[key] = tf.io.parse_tensor(example[key], out_type=tf.uint8)
                example[key] = tf.cast(example[key], dtype=tf.float32)
            else:
                example[key] = tf.io.parse_tensor(example[key], out_type=tf.string)
        return example
    
    def load_dataset(self, files):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self, train_set):
        dataset = self.load_dataset(f"sRGB2XYZ/shards/{train_set}.tfrec")
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset