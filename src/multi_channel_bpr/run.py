"""
Entry point for experiments on Bayesian Personalized Ranking
for Multi-Channel user feedback based on the paper
"Bayesian personalized ranking with multi-channel user feedback."
by Loni, Babak, et al.
in Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
"""
from datetime import datetime
import logging
import os
import pickle
import sys
import copy

from sklearn.model_selection import KFold

from .cli import parse_args
from .model import MultiChannelBPR
from .utils import load_movielens, get_channels


__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Initialize MC-BPR Experiments...")
    train_verbose = True

    train_inter, test_inter, m, n = load_movielens(args.data_path)
    channels = get_channels(train_inter)

    reg_params = {'u': args.reg_param_list[0],
                  'i': args.reg_param_list[1],
                  'j': args.reg_param_list[2]}

    res_dict = {}

    for neg_sampling_mode in args.neg_sampling_modes:
        res_dict[neg_sampling_mode] = {}

        for beta in args.beta_list:
            best_mrr = -1
            res_dict[neg_sampling_mode][beta] = {}

            model = MultiChannelBPR(d=args.d, beta=beta,
                                    rd_seed=args.rd_seed,
                                    channels=channels, n_user=m, n_item=n)
            model.set_data(train_inter, test_inter)
            _logger.info("Training ...")
            for i in range(args.n_epochs):
                res_dict[neg_sampling_mode][beta][i] = {}
                model.fit(lr=args.lr, reg_params=reg_params, n_epochs=1,
                          neg_item_sampling_mode=neg_sampling_mode,
                          verbose=train_verbose)
                _logger.info("Evaluating ...")
                predictions, prec, rec, mrr = model.evaluate(test_inter, args.k)
                res_dict[neg_sampling_mode][beta][i]['map'] = prec
                res_dict[neg_sampling_mode][beta][i]['mar'] = rec
                res_dict[neg_sampling_mode][beta][i]['mrr'] = mrr

                if mrr > best_mrr:
                    best_predictions = copy.deepcopy(predictions)
                    best_mrr = mrr
                    best_model = model
                else:
                    break

    print('finish')
    os.makedirs(args.results_path, exist_ok=True)

    res_filename = 'result.pkl'
    res_filepath = os.path.join(args.results_path, res_filename)
    pickle.dump(res_dict, open(res_filepath, 'wb'))

    predictions_filepath = 'predictions.pkl'
    predictions_filepath = os.path.join(args.results_path, predictions_filepath)
    pickle.dump(best_predictions, open(predictions_filepath, 'wb'))

    model_filepath = 'model.pkl'
    model_filepath = os.path.join(args.results_path, model_filepath)
    pickle.dump(best_model, open(model_filepath, 'wb'))
    print('saved')

    _logger.info("Experiments finished, results saved in %s", res_filepath)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
