import os
import glob
from tqdm.notebook import tqdm
import cv2
import albumentations as A
from sklearn.model_selection import KFold

import tensorflow as tf

class TFRWriter:
    def __init__(self, image_size=256):
        self.main_dir = "sRGB2XYZ"
        self.save_path = os.path.join(self.main_dir, f"shards_{image_size}")
        self.n_splits = 10
        self.image_size = image_size

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
        return image / 255.

    def get_samples(self, train_set):
        path = f"{self.main_dir}/sRGB_{train_set}/*.JPG"
        return glob.glob(path)

    def get_sample_data(self, sample, train_set):
        xyz_filename = sample.replace("sRGB_", "XYZ_").replace(".JPG", ".png")
        assert os.path.exists(xyz_filename), f"{xyz_filename} file does not exist."

        filename = os.path.basename(sample)

        if train_set != "training":
            transform = A.Compose(
                [
                    A.Flip(),
                    A.Resize(self.image_size, self.image_size)
                ],
                additional_targets={'XYZ_image': 'image'})         
        else:
            transform = A.Compose(
                [
                    A.Flip(),
                    A.Resize(self.image_size, self.image_size)
                ],
                additional_targets={'XYZ_image': 'image'}) 
        
        transformed = transform(
            image=self.get_image(sample), 
            XYZ_image=self.get_image(xyz_filename))        

        srgb_image = tf.cast(transformed['image'], dtype=tf.float32)
        xyz_image = tf.cast(transformed['XYZ_image'], dtype=tf.float32)
                
        return [
            tf.io.serialize_tensor(xyz_image, name="XYZ_image"),
            tf.io.serialize_tensor(srgb_image, name="sRGB_image"),
            tf.io.serialize_tensor(filename, name="filename")]
    
    def get_shard_data(self, samples):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        split_list = []
        for _, val in kf.split(samples):
            splits = []
            for i in val:
                splits.append(samples[i])
            split_list.append(splits)
        return split_list

    def write(self):
        sets = ["testing", "validation", "training"]
        for train_set in sets:
            all_samples = self.get_samples(train_set)

            if train_set != "training":
                shard_path = os.path.join(self.save_path, f"{train_set}.tfrec")
                with tf.io.TFRecordWriter(shard_path) as f:
                    for sample in tqdm(all_samples, total=len(all_samples), desc=f"{train_set}"):
                        sample_data = self.get_sample_data(sample, train_set)
                        f.write(self.serialize_example(sample_data))
            else:
                shards = self.get_shard_data(all_samples)
                for shard, samples in tqdm(enumerate(shards), total=self.n_splits, desc=f"{train_set}"):
                    shard_path = os.path.join(self.save_path, f"{train_set}_{shard}.tfrec")
                    with tf.io.TFRecordWriter(shard_path) as f:
                        for sample in samples:
                            sample_data = self.get_sample_data(sample, train_set)
                            f.write(self.serialize_example(sample_data))

class DataLoader:
    def __init__(self, train_set, image_size, batch_size=4):
        self.train_set = train_set
        self.batch_size = batch_size
        self.buffer_size = 64
        self.files = glob.glob(f"sRGB2XYZ/shards_{image_size}/{self.train_set}*.tfrec")
        self.data = self.get_dataset()

    def read_tfrecord(self, example):
        feature_description = {
            'XYZ_image': tf.io.FixedLenFeature([], tf.string),
            'sRGB_image': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string)}
        
        example = tf.io.parse_single_example(example, feature_description)  

        for key in example:
            if key != 'filename':
                example[key] = tf.io.parse_tensor(example[key], out_type=tf.float32)
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

    def get_dataset(self):
        dataset = self.load_dataset(self.files)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(glob.glob(f"sRGB2XYZ/sRGB_{self.train_set}/*")) // self.batch_size