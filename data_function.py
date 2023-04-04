import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn.functional as F # Contains some additional functions such as activations
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from hparam import hparams as hp
import torchio as tio
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RandomGamma,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
    OneHot,
    EnsureShapeMultiple
)
from hparam import hparams as hp
warnings.filterwarnings(action='ignore', category=FutureWarning)

class MedData(torch.utils.data.Dataset):
    def __init__(self, image_dir_IOP, label_dir_IOP,image_dir_Guys, label_dir_Guys, 
                 image_dir_HH, label_dir_HH, train_val_ratio, batch_size, aug_command):
        self.subjects = []
        self.X_IOP_ex = np.load(image_dir_IOP)
        self.y_IOP_ex = np.load(label_dir_IOP)
        self.X_Guys_ex = np.load(image_dir_Guys)
        self.y_Guys_ex = np.load(label_dir_Guys)
        self.X_HH_ex = np.load(image_dir_HH)
        self.y_HH_ex = np.load(label_dir_HH)
        self.train_val_ratio = train_val_ratio
        self.batch_size = batch_size
        self.preprocess = None
        self.transform = None
        self.aug_command = aug_command

    def load_data(self, file_name):
        if file_name == 'IOP':
            X_train = torch.reshape(torch.tensor(self.X_IOP_ex), (self.X_IOP_ex.shape[0], 1, self.X_IOP_ex.shape[1],
                                                                  self.X_IOP_ex.shape[2], self.X_IOP_ex.shape[3]))
            y_train = torch.reshape(torch.tensor(self.y_IOP_ex), (self.y_IOP_ex.shape[0], 1, self.y_IOP_ex.shape[1],
                                                                  self.y_IOP_ex.shape[2], self.y_IOP_ex.shape[3]))
        elif file_name == 'Guys':
            X_train = torch.reshape(torch.tensor(self.X_Guys_ex), (self.X_Guys_ex.shape[0], 1, self.X_Guys_ex.shape[1],
                                                                   self.X_Guys_ex.shape[2], self.X_Guys_ex.shape[3]))
            y_train = torch.reshape(torch.tensor(self.y_Guys_ex), (self.y_Guys_ex.shape[0], 1, self.y_Guys_ex.shape[1],
                                                                   self.y_Guys_ex.shape[2], self.y_Guys_ex.shape[3]))
        elif file_name == 'HH':
            X_train = torch.reshape(torch.tensor(self.X_HH_ex), (self.X_HH_ex.shape[0], 1, self.X_HH_ex.shape[1],
                                                                 self.X_HH_ex.shape[2], self.X_HH_ex.shape[3]))
            y_train = torch.reshape(torch.tensor(self.y_HH_ex), (self.y_HH_ex.shape[0], 1, self.y_HH_ex.shape[1],
                                                                 self.y_HH_ex.shape[2], self.y_HH_ex.shape[3]))

        return X_train, y_train

    def prepare_data(self):
        X_train_IOP, y_train_IOP = self.load_data('IOP')
        for (image, label) in zip(X_train_IOP, y_train_IOP):
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=label),
            )
            self.subjects.append(subject)

        X_train_Guys, y_train_Guys = self.load_data('Guys')
        for (image, label) in zip(X_train_Guys, y_train_Guys):
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=label),
            )
            self.subjects.append(subject)

        X_train_HH, y_train_HH = self.load_data('HH')
        for (image, label) in zip(X_train_HH, y_train_HH):
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=label),
            )
            self.subjects.append(subject)

    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def get_augmentation_transform(self):
        if self.aug_command == 'Affine':
            # apply random affine transformations
            print('********************Do Affine********************')
            auegmentation_transform = RandomAffine()
        elif self.aug_command == 'BiasField':
            # add random smooth intensity variations to the input subjects to simulate different MRI bias field artifacts
            print('********************Do BiasField********************')
            auegmentation_transform = RandomBiasField(p=0.25)
        elif self.aug_command == 'Gamma':
            # apply random gamma correction, help the model handle variations in image contrast
            print('********************Do Gamma********************')
            auegmentation_transform = RandomGamma(p=0.5)
        elif self.aug_command == 'Noise':
            # add random Gaussian noise, helps the model become more robust to noise and artifacts
            print('********************Do Noise********************')
            auegmentation_transform = RandomNoise(p=0.5)
        elif self.aug_command == 'Flip':
            # flip the image along all axes with a probability of 0.5
            print('********************Do Flip********************')
            auegmentation_transform = RandomFlip()
        elif self.aug_command == 'Motion':
            # simulating random rotations and translations, more robust to motion artifacts
            print('********************Do Motion********************')
            auegmentation_transform = RandomMotion(p=0.25)
        elif self.aug_command == 'All':
            print('********************Do All********************')
            auegmentation_transform = Compose([
            #OneOf({
            #    RandomAffine(): 0.8, # Apply a random affine transformation to the image
            #    RandomElasticDeformation(): 0.2,
            #}),
            RandomAffine(),
            RandomBiasField(p=0.25),
            RandomGamma(p=0.5),
            RandomNoise(p=0.5),
            RandomMotion(p=0.25),
            RandomFlip(),
            ])

        return auegmentation_transform
    
    def get_preprocessing_transform(self):
        preprocessing_transform = Compose([
            ToCanonical(),
            Resample(1), # resample the images to a common resolution
            CropOrPad(self.get_max_shape(self.subjects), padding_mode='reflect'), # Modify the field of view by cropping or padding to match a target shape.
            EnsureShapeMultiple(16), #U-Net Ensure that all values in the image shape are divisible by 16
            ZNormalization(), # Subtract mean and divide by standard deviation.
            OneHot(hp.out_channel + 1), # one hot encoding
        ])

        return preprocessing_transform
    
    def setup(self, aug, pre):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio[0]))
        num_val_subjects = int(round(num_subjects * self.train_val_ratio[1]))
        num_test_subjects = num_subjects - num_train_subjects - num_val_subjects

        splits = num_train_subjects, num_val_subjects, num_test_subjects
        train_subjects, val_subjects, test_subjects= random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augmentation = self.get_augmentation_transform()
        self.transform = Compose([self.preprocess, augmentation])
        
        if pre is True:
            print('********************Do Preprocess********************')
            if aug is True:
                print('********************Do Augmentation********************')
                self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
            else:
                print('********************NONO Augmentation********************')
                self.train_set = tio.SubjectsDataset(train_subjects, transform=self.preprocess)
            self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
            self.test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)
        else:
            print('********************NONO Preprocess********************')
            self.train_set = tio.SubjectsDataset(train_subjects, transform=OneHot(hp.out_channel + 1))
            self.val_set = tio.SubjectsDataset(val_subjects, transform=OneHot(hp.out_channel + 1))
            self.test_set = tio.SubjectsDataset(test_subjects, transform=OneHot(hp.out_channel + 1))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, drop_last=True)
