"""
Evaluation module
"""
import logging

import numpy as np

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def score_one_plus_random(k, test_inter, user_reps, item_reps, n_random=1000,
                          verbose=True):
    """
    Computes mean average precision, mean average recall and
    mean reciprocal rank based on the One-plus-random testing methodology
    outlined in
    "Performance of recommender algorithms on top-n recommendation tasks."
    by Cremonesi, Paolo, Yehuda Koren, and Roberto Turrin (2010)

    Args:
        k (int): no. of most relevant items
        test_inter (:obj:`pd.DataFrame`): `M` testing instances (rows)
            with three columns `[user, item, rating]`
        user_reps (dict): representations for all `m` unique users
        item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
        n_random (int): no. of unobserved items to sample randomly
        verbose (bool): verbosity

    Returns:
        prec (float): mean average precision @ k
        rec (float): mean average recall @ k
        mrr (float): mean reciprocal rank @ k
    """
    test_inter_red = test_inter[['user', 'item', 'rating']].values

    n_hits = 0
    rr_agg = 0
    m = test_inter_red.shape[0]
    n_item = item_reps.shape[0]
    all_items = np.array(range(n_item))
    predictions = []

    for idx in range(m):
        u = test_inter_red[idx, 0]
        i = test_inter_red[idx, 1]
        user_embed = user_reps[u]['embed']
        # 1. Randomly select `n_random` unobserved items
        user_item_reps = item_reps[all_items]
        # 2. Predict ratings for test item i and for unobserved items
        user_item_scores = np.dot(user_item_reps, user_embed)
        scores = np.sort(user_item_scores)[::-1][:k + 10]
        user_items = all_items[np.argsort(user_item_scores)[::-1][:k]]
        prediction = all_items[np.argsort(user_item_scores)[::-1][:k + 10]]
        # 3. Get rank p of test item i within rating predictions
        i_idx = np.where(i == user_items)[0]
        if test_inter_red[idx, 2] == 1:
            predictions.append({'user': u, 'correct': i, 'feedback': 'views', 'predictions': prediction, 'scores': scores})
        elif test_inter_red[idx, 2] == 2:
            predictions.append({'user': u, 'correct': i, 'feedback': 'clicks', 'predictions': prediction, 'scores': scores})
        elif test_inter_red[idx, 2] == 3:
            predictions.append({'user': u, 'correct': i, 'feedback': 'favorites', 'predictions': prediction, 'scores': scores})
        elif test_inter_red[idx, 2] == 4:
            predictions.append({'user': u, 'correct': i, 'feedback': 'orders', 'predictions': prediction, 'scores': scores})

        if test_inter_red[idx, 2] in [2, 3, 4]:
            if len(i_idx) != 0:
                # item i is among Top-N predictions
                n_hits += 1
                rr_agg += 1 / (i_idx[0] + 1)

        if verbose and (idx % (m//10) == 0):
            _logger.info("Evaluating %s/%s", str(idx), str(m))

    prec = n_hits / (m*k)
    rec = n_hits / m
    mrr = rr_agg / m

    return predictions, prec, rec, mrr
