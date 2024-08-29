"""
Helper functions
"""
import logging
import os
import pdb

import numpy as np
import pandas as pd

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def get_pos_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Returns the sampling distribution for positive
    feedback channels L using either a `non-uniform` or `uniform` approach

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            positive feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            positive feedback channel
        mode (str): either `uniform` meaning all positive levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of positive levels

    Returns:
        dist (dict): positive channel sampling distribution
    """
    if mode == 'non-uniform':
        nominators = weights * level_counts
        denominator = sum(nominators)
        dist = nominators / denominator
    else:
        n_levels = len(weights)
        dist = np.ones(n_levels) / n_levels

    dist = dict(zip(list(weights), dist))

    return dist


def get_neg_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Compute negative feedback channel distribution

    Args:
        weights (:obj:`np.array`): (w, ) `w` rating values representing distinct
            negative feedback channels
        level_counts (:obj:`np.array`): (s, ) count `s` of ratings for each
            negative feedback channel
        mode: either `uniform` meaning all negative levels are
            equally relevant or `non-uniform` which imposes
            a (rating*count)-weighted distribution of negative levels

    Returns:
        dist (dict): negative channel sampling distribution
    """
    if mode == 'non-uniform':
        nominators = [weight * count for weight, count in zip(weights, level_counts)]
        denominator = sum(nominators)
        if denominator != 0:
            dist = list(nom / denominator for nom in nominators)
        else:
            dist = [0] * len(nominators)
    else:
        n_levels = len(weights)
        if n_levels != 0:
            dist = [1 / n_levels] * n_levels
        else:
            dist = []

    if np.abs(np.sum(dist)-1) > 0.00001:
        _logger.warning("Dist sum unequal 1.")

    dist = dict(zip(list(weights), dist))

    return dist


def rms(x):
    """
    Calculates Root Mean Square for array x
    """
    s = np.square(x)
    ms = np.mean(s)
    result = np.sqrt(ms)

    return result


def sigmoid(x):
    """
    Calculates Sigmoid of x
    """
    return 1/(1+np.exp(-x))


def load_movielens(path):
    """
    loads the movielens 1M dataset, ignoring temporal information

    Args:
        path (str): path pointing to folder with interaction data `ratings.dat`

    Returns:
        ratings (:obj:`pd.DataFrame`): overall interaction instances (rows)
            with three columns `[user, item, rating]`
        m (int): no. of unique users in the dataset
        n (int): no. of unique items in the dataset
    """
    print(os.path.join(path, 'train.parquet'))
    train_ratings = pd.read_parquet(os.path.join(path, 'train.parquet'))
    test_ratings = pd.read_parquet(os.path.join(path, 'test.parquet'))

    m = pd.concat([train_ratings['user'], test_ratings['user']]).unique().shape[0]
    n = pd.concat([train_ratings['item'], test_ratings['item']]).unique().shape[0]

    train_ratings['user'] = train_ratings['user'].astype(int)
    test_ratings['user'] = test_ratings['user'].astype(int)

    return train_ratings, test_ratings, m, n


def get_channels(inter_df):
    """
    Return existing feedback channels ordered by descending preference level

    Args:
        inter_df (:obj:`pd.DataFrame`): overall interaction instances (rows)
            with three columns `[user, item, rating]`
    Returns:
        channels ([int]): rating values representing distinct feedback channels
    """
    channels = list(inter_df['rating'].unique())
    channels.sort()

    return channels[::-1]
