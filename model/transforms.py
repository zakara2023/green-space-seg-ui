"""
transforms.py

The image and geo location transforms for AerMAE networks
"""
from datetime import datetime

import math
import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2 as T
from PIL.Image import Image
from pycocotools import mask


class GeoLocation:
    """Transform to get the geo location for geopandas json data."""

    def __call__(self, meta: dict) -> torch.Tensor:
        """Take the location from the geopandas dictionary."""
        feat = meta['features'][0]
        #loc = feat['geometry']['coordinates'][0][0] # grab the polygon box from the geopandas json
        #loc = (np.array(loc) + 180)# / 360 # normalize to [0, 1] min-max is (-180, 180) hence loc - (-180) / (-180 - 180) => loc + 180 / 360
        tile = feat['properties']['tile_id']
        loc = np.array([int(tile[1:3]), int(tile[4:])])
        return torch.from_numpy(loc).float()


class GeoLocationAndDate:
    """Transform to get the geo location for geopandas json data."""

    def __call__(self, meta: dict) -> torch.Tensor:
        """Take the location from the geopandas dictionary."""
        feat = meta['features'][0]

        coord = feat['geometry']['coordinates'][0][0] # grab the polygon box from the geopandas json
        lon, lat = np.array(coord) // 1 # normalize to [-1, 1]
        #lat_grid = lat / 90
        #lon_grid = lon / 180
        #lat_frac = lat % 1
        #lon_frac = lon % 1
        #tile = feat['properties']['tile_id']
        #scale = feat['properties']['scale']

        altitude = feat['properties']['altitude']
        if math.isnan(altitude):
            altitude = 0
        else:
            altitude = max(altitude, 0) / 1000

        date = feat['properties']['Date']
        date = datetime.strptime(date, "%Y/%m/%d")

        meta = np.array([
            lat,
            #lat_frac,
            lon,
            #lon_frac,
            #altitude,
            #date.month,
            #date.year - 1937 # earliest year captured
        ])

        return torch.from_numpy(meta).float()


class AerMAETransforms:
    """Aer MAE image and metadata transformations."""

    def __init__(self, is_train: bool, mu: list[float], sigma: list[float]):
        """
        :param is_train: Flag indicating if transforms are for training
        :type is_train: bool
        :param mu: The mean of the images
        :type mu: list[float]
        :param sigma: The standard deviation of the images
        :type sigma: list[float]
        """
        self.img_transforms = build_img_transforms(is_train)
        self.img_post_transforms = build_post_img_transforms(mu, sigma)
        self.meta_transforms = build_meta_transforms()

    def __call__(self, sample) -> tuple:
        """Apply the transformations."""
        img, meta = sample
        img = self.img_transforms(img)
        x = self.img_post_transforms(img)
        meta = self.meta_transforms(meta)
        return x, meta


class AeGISTransforms:
    """Aer GIS image and segmentation transformations."""

    def __init__(self, is_train: bool, mu: list[float], sigma: list[float]):
        """
        :param is_train: Flag indicating if transforms are for training
        :type is_train: bool
        :param mu: The mean of the images
        :type mu: list[float]
        :param sigma: The standard deviation of the images
        :type sigma: list[float]
        """
        self.img_transforms = build_img_transforms(is_train)
        self.img_post_transforms = build_post_img_transforms(mu, sigma)
        self.segmentation_transforms = build_segmentation_transforms()

    def __call__(self, sample) -> tuple:
        """Apply the transformations."""
        img, meta = sample
        img = self.img_transforms(img)
        img = self.img_post_transforms(img)

        feat = meta['features'][0]
        props = feat['properties']
        segmentation = mask.decode(props['segmentation'])
        segmentation = segmentation.astype(np.float32)

        coords = feat['geometry']['coordinates'][0]
        lon, lat = np.array(coords[0]) // 1
        meta = np.array([lat, lon])
        meta = torch.from_numpy(meta).float()

        return img, meta, segmentation


def build_segmentation_transforms() -> callable:
    """Build the segmentation data transformations.

    :return: The transforms
    :rtype: callable
    """
    def _transforms(meta: dict):
        props = meta['features'][0]['properties']
        segmentation = mask.decode(props['segmentation'])
        segmentation = segmentation.astype(np.float32)
        return segmentation

    return _transforms


def build_meta_transforms() -> callable:
    """Build the meta data transformations.

    :return: The transforms
    :rtype: callable
    """
    return transforms.Compose([GeoLocationAndDate()])


def build_post_img_transforms(mu: list[float], sigma: list[float]) -> callable:
    return T.Compose([T.Normalize(mu, sigma)])


def build_img_transforms(is_train: bool) -> callable:
    """Build the image transformations.

    :param is_train: Flag indicating if transforms are for training
    :type is_train: bool
    :param mu: The mean of the images
    :type mu: list[float]
    :param sigma: The standard deviation of the images
    :type sigma: list[float]
    :return: The transforms
    :rtype: callable
    """
    base_tr = []
    base_tr.append(T.ToImage())
    base_tr.append(T.ToDtype(torch.uint8, scale=True))
    base_tr.append(T.Grayscale())
    base_tr.append(T.Resize(224, interpolation=T.InterpolationMode.BICUBIC))

    if is_train:
        #base_tr.append(T.ColorJitter(brightness=(0.5, 1.5)))
        #base_tr.append(T.RandomCrop(224))
        #base_tr.append(T.RandomHorizontalFlip())
        #base_tr.append(T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC))
        base_tr.append(T.CenterCrop(224))
    else:
        base_tr.append(T.CenterCrop(224))

    base_tr.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(base_tr)


def denormalize(img: Image, mu: list[float], sigma: list[float]) -> Image:
    """Return the denormalized version of the image"""
    return img * np.array(sigma) + np.array(mu)


def normalize(img: Image, mu: list[float], sigma: list[float]) -> Image:
    """Return the normalized version of the image"""
    return F.normalize(img, mu, sigma)


def normalize_patches(images: torch.Tensor, eps: float=1.e-6) -> torch.Tensor:
    """Normalize image patches with their own mu and sigma"""
    mu = images.mean(dim=-1, keepdim=True)
    var = images.var(dim=-1, keepdim=True)
    return (images - mu) / (var + eps)**.5


def patch_images(images: torch.Tensor, patch_size: int=16) -> torch.Tensor:
    """Patch the image tensors"""
    return (images
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
        .flatten(2, 3)
        .flatten(3, 4)
        .squeeze(1))


def unpatch_images(images: torch.Tensor, patch_size: int=16) -> torch.Tensor:
    """Unpatch the image tensors"""
    h = w = int(images.shape[1]**.5)
    return (images.reshape(images.shape[0], h, w, patch_size, patch_size)
        .transpose(2, 3)
        .flatten(1, 2)
        .flatten(2, 3))
