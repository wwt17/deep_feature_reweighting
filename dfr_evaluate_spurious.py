"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy as sp
import os
import tqdm
import argparse
import sys
from pathlib import Path
from collections import defaultdict, namedtuple
import json
from functools import partial
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from bayesian_models import LabelRegressionModel, DirichletObservationModel


OPTION_SET_NAME = "WaterBirds"
PENALTY_OPTIONS = ["l1", "l2"][:1]
if OPTION_SET_NAME == "WaterBirds":
    C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
    CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
elif OPTION_SET_NAME == "CelebA":
    C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
    CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 500.]
CLASS_WEIGHT_OPTIONS = (
    [(1., w) for w in CLASS_WEIGHT_OPTIONS] +
    [(w, 1.) for w in CLASS_WEIGHT_OPTIONS if w != 1.])
DEFAULT_CLASS_WEIGHT = [(1., 1.)]
INTERCEPT_SCALING_OPTIONS = [1., 10., 30., 100.]


class HyperParams(
    namedtuple(
        "HyperParams_",
        ["penalty", "C", "intercept_scaling", "class_weight"]
    )
):
    """Hyper-parameters for LogisticRegression.
    class_weight should be a tuple.
    """
    def to_kwargs(self):
        kwargs = self._asdict()
        kwargs["class_weight"] = {
            i: w for i, w in enumerate(kwargs["class_weight"])}
        return kwargs

    def __str__(self):
        return f"{self.penalty},C={self.C:<4},IS={self.intercept_scaling:<5},CW={str(self.class_weight):<13}"


BL_PRECISION_OPTIONS = [300., 1000., 3000.]
BL_NOISE_PRECISION_OPTIONS = [10., 30., 100., 300.]
BL_INTERCEPT_SCALING_OPTIONS = [100.]


class BayesianHyperParams(
    namedtuple(
        "BayesianHyperParams_",
        ["precision", "noise_precision", "intercept_scaling"]
    )
):
    """Hyper-parameters for BayesianLinearRegression.
    """
    def to_kwargs(self):
        return self._asdict()
    
    def __str__(self):
        return f"prior={self.precision:<6},noise={self.noise_precision:<5},IS={self.intercept_scaling:<5}"


INDENT = '\t'
np.set_printoptions(precision=4, linewidth=100)


def build_argparser():
    parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Train dataset directory")
    parser.add_argument(
        "--result_path", type=Path, default=Path("logs/"),
        help="Path to save results")
    parser.add_argument(
        "--ckpt_path", type=Path, default=None, help="Checkpoint path")
    parser.add_argument(
        "--expr", type=str, nargs="*",
        choices=["base", "on_val", "on_train", "on_unbalanced_train", "blreg_on_val", "blreg_on_unbalanced_train", "dir_on_val"],
        default=["base", "on_val", "on_train"],
        help="Experiments to run")
    parser.add_argument(
        "--train_frac", type=float, default=1.,
        help="Fraction of train set to subsample")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Batch size")
    parser.add_argument(
        "--balance_dfr_val", type=bool, default=True, required=False,
        help="Subset validation to have equal groups for DFR(Val)")
    parser.add_argument(
        "--notrain_dfr_val", type=bool, default=True, required=False,
        help="Do not add train data for DFR(Val)")
    parser.add_argument(
        "--tune_class_weights_dfr_train", action='store_true',
        help="Learn class weights for DFR(Train)")
    parser.add_argument(
        "--penalty", type=lambda s: None if s == "None" else s,
        choices=["tune", "l1", "l2", "elasticnet", None], default="tune",
        help="regularization (i.e., penalty) for logistic regression.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed")
    return parser


def concatenate_datasets(*datasets):
    return tuple(np.concatenate(field) for field in zip(*datasets))


def group_balanced(dataset, n_groups):
    x, y, g = dataset
    g_idxs = [np.where(g == g_id)[0] for g_id in range(n_groups)]
    min_g_size = np.min([len(g_idx) for g_idx in g_idxs])
    for g_idx in g_idxs:
        np.random.shuffle(g_idx)
    idx = np.concatenate([g_idx[:min_g_size] for g_idx in g_idxs])
    return x[idx], y[idx], g[idx]


def get_split(split, all_embeddings, all_y, all_g, n_groups=None, group_balance=False, max_n=None, random_selection=False):
    """Get the original split.
    Returns: (x_split, y_split, g_split)
    """
    x = all_embeddings[split]
    y = all_y[split]
    g = all_g[split]

    if max_n is not None:
        if random_selection:
            idx = np.arange(len(x))
            np.random.shuffle(idx)
            idx = idx[:max_n]
        else:
            idx = np.s_[:max_n]
        x, y, g = x[idx], y[idx], g[idx]

    if group_balance:
        x, y, g = group_balanced((x, y, g), n_groups)

    return x, y, g


def get_val_set(all_embeddings, all_y, all_g, n_groups, group_balance=False, add_train=True, random_selection=True):
    x, y, g = get_split("val", all_embeddings, all_y, all_g, n_groups, group_balance=group_balance)
    if add_train:
        x, y, g = concatenate_datasets(
            get_split("train", all_embeddings, all_y, all_g,
                      max_n=len(x), random_selection=random_selection),
            (x, y, g))
    return x, y, g


def get_train_val_set(all_embeddings, all_y, all_g, n_groups, group_balance=False, max_n=None, random_selection=False):
    """Get original train and val sets.
    Returns: ((x_train, y_train, g_train), (x_val, y_val, g_val))
    """
    return (
        get_split(
            "train", all_embeddings, all_y, all_g, n_groups,
            group_balance=group_balance,
            max_n=max_n,
            random_selection=random_selection),
        get_split(
            "val", all_embeddings, all_y, all_g, n_groups,
            group_balance=False)
    )


def split_val_set(all_embeddings, all_y, all_g, n_groups, n_val=None, group_balance=False, add_train=True):
    """Split a valtrain set from the val set and optionally merge with part of the train set.
    Args:
        n_val: Number of remaining val examples. Default: len(val_set) // 2
        group_balance: Whether to make the valtrain set group balanced.
        add_train: Whether to add part of the train set.
    Returns: ((x_train, y_train, g_train), (x_val, y_val, g_val))
    """
    x_val = all_embeddings["val"]
    y_val = all_y["val"]
    g_val = all_g["val"]

    if n_val is None:
        n_val = len(x_val) // 2

    # randomly split val set
    idx = np.arange(len(x_val))
    np.random.shuffle(idx)
    x_train = x_val[idx[n_val:]]
    y_train = y_val[idx[n_val:]]
    g_train = g_val[idx[n_val:]]
    x_val = x_val[idx[:n_val]]
    y_val = y_val[idx[:n_val]]
    g_val = g_val[idx[:n_val]]

    if group_balance:
        x_train, y_train, g_train = group_balanced((x_train, y_train, g_train), n_groups)

    if add_train:
        x_train, y_train, g_train = concatenate_datasets(
            get_split("train", all_embeddings, all_y, all_g,
                      max_n=len(x_train), random_selection=False),
            (x_train, y_train, g_train))

    return (x_train, y_train, g_train), (x_val, y_val, g_val)


def build_logistic_regression_model(
        hypers, d=None, logreg_kwargs=dict(solver="liblinear")):
    return LogisticRegression(
        **hypers.to_kwargs(), **logreg_kwargs)


def build_label_regression_model(hypers, d):
    return LabelRegressionModel(d, **hypers.to_kwargs())


def build_dirichlet_observation_model(hypers, d, num_classes=2):
    return DirichletObservationModel(num_classes, d, **hypers.to_kwargs())


def get_conf_acc(logits, target):
    probs = sp.special.softmax(logits, axis=-1)
    pred_ = np.argmax(probs, axis=-1, keepdims=True)
    pred = pred_.squeeze(axis=-1)
    conf = np.take_along_axis(probs, pred_, axis=-1).squeeze(axis=-1)
    acc = pred == target
    return conf, acc


def get_ece(conf, acc, n_bins=10, verbose=True):
    assert len(conf) == len(acc)
    assert np.all((conf > 0) & (conf <= 1))
    bin_counts = np.ndarray((n_bins,), dtype=int)
    sum_conf = np.ndarray((n_bins,), dtype=float)
    sum_acc = np.ndarray((n_bins,), dtype=int)
    for i_bin in range(n_bins):
        a, b = i_bin / n_bins, (i_bin + 1) / n_bins
        subsamples = (conf > a) & (conf <= b)
        bin_counts[i_bin] = subsamples.sum()
        sum_conf[i_bin] = conf[subsamples].sum()
        sum_acc[i_bin] = acc[subsamples].sum()
    sum_overconf = sum_conf - sum_acc
    ece = np.abs(sum_overconf).sum() / len(conf)
    if verbose:
        print(f"ECE={ece:.4f}")
        overconf = sum_overconf / bin_counts
        bin_low = n_bins // 2
        print("bin_counts=" + " ".join(map("{:6d}".format, bin_counts[bin_low:])))
        print("  overconf=" + " ".join(map("{:6.3f}".format, overconf[bin_low:])))
    return ece


def evaluate_on_dataset(
        pred_probs, dataset, n_groups, with_ece=False, n_bins=10,
        verbose=True):
    """Evaluate model on dataset.
    Args:
        pred_probs: predicted probabilities of shape (n, n_classes)
        dataset: (x, y, g)
        with_ece: bool, whether to compute ECE. If False, returned
            ece, group_eces will both be None
        n_bins: int, number of bins in computing ECE
        verbose: bool, whether to print evaluation information
    Returns:
        preds, corrects, group_accs, ece, group_eces
    """
    x, y, g = dataset
    preds = pred_probs.argmax(axis=-1)
    corrects = preds == y
    group_accs = [corrects[g == g_id].mean() for g_id in range(n_groups)]

    if with_ece:
        conf = np.take_along_axis(pred_probs, np.expand_dims(preds, axis=-1), axis=-1).squeeze(axis=-1)  # confidence
        ece = get_ece(conf, corrects, n_bins=n_bins, verbose=verbose)
        group_eces = [
            get_ece(conf[g == g_id], corrects[g == g_id], n_bins=n_bins,
                    verbose=verbose)
            for g_id in range(n_groups)]
    else:
        ece, group_eces = None, None

    return preds, corrects, group_accs, ece, group_eces


def worst_group_objective(val_preds, corrects, group_accs, ece=None, group_eces=None, ece_ratio=0.):
    """
    Args:
        ece_ratio: optimization objective is minimizing
            (1-ece_ratio) * worst_group_err + ece_ratio * worst_group_ece
    Returns:
        objective, info_str
    """
    group_accs = np.array(group_accs)
    worst_group_acc = np.min(group_accs)
    worst_group_err = 1 - worst_group_acc
    if group_eces is not None:
        worst_group_ece = np.max(group_eces)
    else:
        worst_group_ece = 0.
    objective = (1-ece_ratio) * worst_group_err + ece_ratio * worst_group_ece

    info_str = f"accs={group_accs} worst={worst_group_acc:.4f}"
    if ece is not None:
        info_str += f" {ece=:.4f}"
    if group_eces is not None:
        info_str += f" worst={worst_group_ece:.4f}"
    info_str += f" obj={objective:.4f}"

    return objective, info_str


def tune(
        hyper_options, get_datasets, n_groups, scaler="train", num_retrains=1,
        build_model=build_logistic_regression_model,
        objective=worst_group_objective,
        with_ece=False,
        verbose=False):
    """Tune hyperparameters of the model to minimize the objective.
    Args:
        hyper_options: iterable of HyperParams, hyper-parameter options to try.
        get_datasets: callable to get train and val sets.
        n_groups: int, number of groups
        scaler: If set to "train", fit the train set each time from get_datasets. None for no preprocessing.
        num_retrains: int, number of calls to get_datasets()
        build_model: Callable, model(hypers, d) builds the model with hypers
            and dimension d.
            model.fit(x_train, y_train) train the model, and
            model.predict_proba(x_val) returns predictive probabilities over
            classes on x_val.
        objective: Callable, objective(*eval_result) returns
            objective, info_str which are the objective value and the
            infomation string to print.
        with_ece: bool, whether to compute ECE.
        verbose: bool, whether to print evaluation information.
    """
    objectives = defaultdict(float)
    for i in range(num_retrains):
        (x_train, y_train, g_train), (x_val, y_val, g_val) = get_datasets()
        print(f"train group sizes: {np.bincount(g_train)}")
        if scaler == "train":
            _scaler = StandardScaler()
            _scaler.fit(x_train)
        else:
            _scaler = scaler
        if _scaler is not None:
            x_train = _scaler.transform(x_train)
            x_val = _scaler.transform(x_val)

        for hypers in hyper_options:
            model = build_model(hypers, x_train.shape[-1])
            model.fit(x_train, y_train)
            val_pred_probs = model.predict_proba(x_val)
            eval_result = evaluate_on_dataset(
                val_pred_probs, (x_val, y_val, g_val), n_groups,
                with_ece=with_ece, verbose=verbose)
            obj, info_str = objective(*eval_result)
            objectives[hypers] += obj
            print(f"{hypers} {info_str}")

    ks, vs = list(objectives.keys()), list(objectives.values())
    best_hypers = ks[np.argmin(vs)]
    return best_hypers


def eval(model, datasets, verbose=True):
    results = {}
    for split, dataset in datasets.items():
        x, y, g = dataset
        pred_probs = model.predict_proba(x)
        preds, corrects, group_accs, ece, group_eces = evaluate_on_dataset(
            pred_probs, dataset, n_groups, with_ece=True, verbose=verbose)
        mean_acc = corrects.mean()
        worst_group_acc = np.min(group_accs)
        results[split] = {
            "mean_acc": mean_acc,
            "group_accs": group_accs,
            "worst_group_acc": worst_group_acc,
            "ece": ece,
            "group_eces": group_eces,
        }

    return results


def print_logreg(logreg):
    coef_mean, coef_var = np.mean(logreg.coef_), np.var(logreg.coef_)
    coef_l1_norm = np.linalg.norm(logreg.coef_, ord=1, axis=-1)
    coef_l2_norm = np.linalg.norm(logreg.coef_, ord=2, axis=-1)
    print(f"coef: mean={coef_mean} var={coef_var} l1={coef_l1_norm} l2={coef_l2_norm}\n{logreg.coef_}")
    print(f"intercept: {logreg.intercept_}")


def dfr_eval(
        hypers, get_train_dataset, get_eval_dataset, n_groups, scaler,
        num_retrains=20,
        logreg_kwargs=dict(solver="liblinear"),
        verbose=True):
    coefs, intercepts = [], []

    for i in range(num_retrains):
        x_train, y_train, g_train = get_train_dataset()
        print(f"train group sizes: {np.bincount(g_train)}")
        if scaler is not None:
            x_train = scaler.transform(x_train)

        logreg = build_logistic_regression_model(
            hypers, logreg_kwargs=logreg_kwargs)
        logreg.fit(x_train, y_train)

        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)
        if verbose:
            print(f"logreg #{i}:")
            print_logreg(logreg)

    x_test, y_test, g_test = get_eval_dataset()
    print(f"test group sizes: {np.bincount(g_test)}")
    if scaler is not None:
        x_test = scaler.transform(x_test)

    logreg = build_logistic_regression_model(
        hypers, logreg_kwargs=logreg_kwargs)
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    if verbose:
        print(f"logreg averaged:")
        print_logreg(logreg)

    datasets = {
        "test": (x_test, y_test, g_test),
        "train": (x_train, y_train, g_train),
    }
    return eval(logreg, datasets, verbose=verbose)


def train_once_eval(
        hypers, get_train_dataset, get_eval_dataset, n_groups, scaler,
        build_model=build_label_regression_model,
        verbose=True):
    x_train, y_train, g_train = get_train_dataset()
    print(f"train group sizes: {np.bincount(g_train)}")
    if scaler is not None:
        x_train = scaler.transform(x_train)

    model = build_model(hypers, x_train.shape[-1])
    model.fit(x_train, y_train)

    x_test, y_test, g_test = get_eval_dataset()
    print(f"test group sizes: {np.bincount(g_test)}")
    if scaler is not None:
        x_test = scaler.transform(x_test)

    datasets = {
        "test": (x_test, y_test, g_test),
        "train": (x_train, y_train, g_train),
    }
    return eval(model, datasets, verbose=verbose)


if __name__ == '__main__':
    argparser = build_argparser()
    args = argparser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    ## Load data
    target_resolution = (224, 224)
    train_transform = get_transform_cub(
        target_resolution=target_resolution,
        train=True, augment_data=False)
    test_transform = get_transform_cub(
        target_resolution=target_resolution,
        train=False, augment_data=False)

    datasets = {
        split: WaterBirdsDataset(
            basedir=args.data_dir, split=split, transform=test_transform)  # always use test_transform for evaluation
        for split in ["train", "val", "test"]
    }
    trainset, valset, testset = datasets.values()
    n_groups = trainset.n_groups

    loader_kwargs = {'batch_size': args.batch_size,
                    'num_workers': 4, 'pin_memory': True,
                    "reweight_places": None}
    loaders = {
        split: get_loader(
            dataset, train=False, reweight_groups=None, reweight_classes=None,  # always in test mode for evaluation
            # train=True, reweight_groups=False, reweight_classes=False,
            **loader_kwargs)
        for split, dataset in datasets.items()
    }

    # Load model
    n_classes = trainset.n_classes
    model = torchvision.models.resnet50(weights=None)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.cuda()
    model.eval()

    all_results = {}

    # Extract embeddings
    def get_embed(m, x):
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


    all_embeddings = {}
    all_y, all_p, all_g = {}, {}, {}
    for name, loader in loaders.items():
        all_embeddings[name] = []
        all_y[name], all_p[name], all_g[name] = [], [], []
        for x, y, g, p in tqdm.tqdm(loader):
            with torch.no_grad():
                all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
                all_y[name].append(y.detach().cpu().numpy())
                all_g[name].append(g.detach().cpu().numpy())
                all_p[name].append(p.detach().cpu().numpy())
        all_embeddings[name] = np.vstack(all_embeddings[name])
        all_y[name] = np.concatenate(all_y[name])
        all_g[name] = np.concatenate(all_g[name])
        all_p[name] = np.concatenate(all_p[name])

    scaler = StandardScaler()
    scaler.fit(all_embeddings["train"])

    # Hyperparams options
    penalty_options = [args.penalty] if args.penalty != "tune" else PENALTY_OPTIONS

    for expr in args.expr:
        if expr == "base":  # Evaluate base model
            expr_desc = "Base Model"
            print(expr_desc)
            results_name = "base_model_results"
            get_yp_func = partial(get_y_p, n_places=trainset.n_places)
            results = {}
            for split, loader in loaders.items():
                split_results, logits = evaluate(model, loader, get_yp_func, return_logits=True)
                y, g = loader.dataset.y_array, loader.dataset.group_array
                conf, acc = get_conf_acc(logits, y)
                mean_accuracy = acc.mean()
                np.testing.assert_approx_equal(mean_accuracy, split_results["mean_accuracy"])
                ece = get_ece(conf, acc)
                group_eces = [
                    get_ece(conf[g == g_id], acc[g == g_id])
                    for g_id in range(n_groups)]
                split_results["calibration"] = {
                    "ece": ece,
                    "group_eces": group_eces,
                }
                results[split] = split_results
            model.eval()

        elif expr == "on_val":  # DFR on validation
            expr_desc = "DFR on validation"
            print(expr_desc)
            results_name = "dfr_val_results"
            results = {}
            learn_class_weights = not(args.balance_dfr_val and args.notrain_dfr_val)
            class_weight_options = (
                CLASS_WEIGHT_OPTIONS if learn_class_weights else
                DEFAULT_CLASS_WEIGHT)
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, C_OPTIONS, INTERCEPT_SCALING_OPTIONS,
                        class_weight_options),
            )
            hyper = tune(
                hyper_options,
                partial(
                    split_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=args.balance_dfr_val,
                    add_train=not args.notrain_dfr_val
                ),
                n_groups,
                build_model=build_logistic_regression_model
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(dfr_eval(
                hyper,
                partial(
                    get_val_set,
                    all_embeddings, all_y, all_g, n_groups,
                    group_balance=args.balance_dfr_val,
                    add_train=not args.notrain_dfr_val, random_selection=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler
            ))

        elif expr == "on_train":  # DFR on train subsampled
            expr_desc = "DFR on train subsampled"
            print(expr_desc)
            results_name = "dfr_train_results"
            results = {}
            learn_class_weights = args.tune_class_weights_dfr_train
            class_weight_options = (
                CLASS_WEIGHT_OPTIONS if learn_class_weights else
                DEFAULT_CLASS_WEIGHT)
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, C_OPTIONS, INTERCEPT_SCALING_OPTIONS,
                        class_weight_options),
            )
            hyper = tune(
                hyper_options,
                partial(
                    get_train_val_set,
                    all_embeddings, all_y, all_g, n_groups,
                    group_balance=True
                ),
                n_groups, scaler=scaler,
                build_model=partial(
                    build_logistic_regression_model,
                    logreg_kwargs=dict(solver="liblinear", max_iter=20)
                )
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(dfr_eval(
                hyper,
                partial(
                    get_split, "train", all_embeddings, all_y, all_g, n_groups,
                    group_balance=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler
            ))

        elif expr == "on_unbalanced_train":  # DFR on unbalanced subsampled train
            n_train = len(all_embeddings["train"])
            max_n = int(n_train * args.train_frac)
            expr_desc = f"DFR on unbalanced train ({args.train_frac:.2%}={max_n}/{n_train})"
            print(expr_desc)
            results_name = "dfr_unbalanced_train_results"
            results = {}
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, C_OPTIONS, INTERCEPT_SCALING_OPTIONS,
                        class_weight_options),
            )
            hyper = tune(
                hyper_options,
                partial(
                    get_train_val_set,
                    all_embeddings, all_y, all_g, n_groups,
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                n_groups, scaler=scaler,
                build_model=partial(
                    build_logistic_regression_model,
                    logreg_kwargs=dict(solver="liblinear", max_iter=20)
                )
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(dfr_eval(
                hyper,
                partial(
                    get_split, "train", all_embeddings, all_y, all_g, n_groups,
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler
            ))

        elif expr == "blreg_on_val":  # Bayesian Linear Regression on Labels
            expr_desc = "Bayesian Linear Regression on Labels on validation"
            print(expr_desc)
            results_name = "blreg_on_val_results"
            results = {}
            hyper_options = map(
                BayesianHyperParams._make,
                product(BL_PRECISION_OPTIONS, BL_NOISE_PRECISION_OPTIONS,
                        BL_INTERCEPT_SCALING_OPTIONS)
            )
            hyper = tune(
                hyper_options,
                partial(
                    split_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=True,
                    add_train=False
                ),
                n_groups,
                scaler=scaler,
                build_model=build_label_regression_model,
                objective=partial(worst_group_objective, ece_ratio=.5),
                with_ece=True,
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(train_once_eval(
                hyper,
                partial(
                    get_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=True,
                    add_train=False,
                    random_selection=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler,
                build_model=build_label_regression_model,
            ))

        elif expr == "blreg_on_unbalanced_train":  # Bayesian Linear Regression on unbalanced subsampled train
            n_train = len(all_embeddings["train"])
            max_n = int(n_train * args.train_frac)
            expr_desc = f"Bayesian Linear Regression on Labels on unbalanced train ({args.train_frac:.2%}={max_n}/{n_train})"
            print(expr_desc)
            results_name = "blreg_on_unbalanced_train_results"
            results = {}
            hyper_options = map(
                BayesianHyperParams._make,
                product(BL_PRECISION_OPTIONS, BL_NOISE_PRECISION_OPTIONS,
                        BL_INTERCEPT_SCALING_OPTIONS)
            )
            hyper = tune(
                hyper_options,
                partial(
                    get_train_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                n_groups,
                scaler=scaler,
                build_model=build_label_regression_model,
                objective=partial(worst_group_objective, ece_ratio=.5),
                with_ece=True,
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(train_once_eval(
                hyper,
                partial(
                    get_split, "train", all_embeddings, all_y, all_g, n_groups,
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler,
                build_model=build_label_regression_model,
            ))

        elif expr == "dir_on_val":  # Dirichlet model
            expr_desc = "Dirichlet Observation Model on validation"
            print(expr_desc)
            results_name = "dir_on_val_results"
            results = {}
            hyper_options = map(
                BayesianHyperParams._make,
                product(BL_PRECISION_OPTIONS, BL_NOISE_PRECISION_OPTIONS,
                        [0.])
            )
            hyper = tune(
                hyper_options,
                partial(
                    split_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=True,
                    add_train=False
                ),
                n_groups,
                scaler=scaler,
                build_model=build_dirichlet_observation_model,
                objective=partial(worst_group_objective, ece_ratio=.5),
                with_ece=True,
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(train_once_eval(
                hyper,
                partial(
                    get_val_set, all_embeddings, all_y, all_g, n_groups,
                    group_balance=True,
                    add_train=False,
                    random_selection=True
                ),
                partial(
                    get_split, "test", all_embeddings, all_y, all_g
                ),
                n_groups, scaler,
                build_model=build_dirichlet_observation_model,
            ))

        print(expr_desc+" results:")
        print(json.dumps(results, indent=INDENT))
        print()
        all_results[results_name] = results

        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_path, 'w') as f:
            json.dump(all_results, f, indent=INDENT)
