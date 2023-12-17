"""Evaluate DFR on spurious correlations datasets."""

from typing import Optional

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
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
from itertools import product, chain
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data, concatenate_datasets
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p, SoftmaxClassifier


def get_n_rows(n, n_cols):
    return (n - 1) // n_cols + 1


def reciprocal_class_weights(class_weights):
    class_weights = sorted(class_weights)
    return (
        [(w, 1.) for w in reversed(class_weights)] +
        [(1., w) for w in class_weights if w != 1.]
    )


PENALTY_OPTIONS = ["l1", "l2"][:1]
DATASET_OPTIONS = {
    "waterbird": {
        "C": [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01],
        "intercept_scaling": [1., 10., 30., 100.],
        "class_weight": reciprocal_class_weights([1., 2., 3., 10., 100., 300., 1000.]),
    },
    "celeba": {
        "C": [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003],
        "intercept_scaling": [1., 10., 30., 100.],
        "class_weight": [(1., w) for w in [1., 2., 3., 10., 100., 300., 500.]],
    },
}
DEFAULT_CLASS_WEIGHT = [(1., 1.)]


def get_dataset_name(data_dir):
    data_dir = str(data_dir)
    for dataset_name in DATASET_OPTIONS.keys():
        if dataset_name in data_dir:
            return dataset_name
    return None


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
        choices=["base", "on_val", "on_train", "on_unbalanced_train"],
        default=["base", "on_val", "on_train"],
        help="Experiments to run")
    parser.add_argument(
        "--original_base_eval", action="store_true",
        help="Use original evaluation for the base model")
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


def group_balanced(dataset):
    g_idxs = [np.where(dataset.g == g_id)[0] for g_id in range(dataset.n_groups)]
    min_g_size = np.min([len(g_idx) for g_idx in g_idxs])
    for g_idx in g_idxs:
        np.random.shuffle(g_idx)
    idx = np.concatenate([g_idx[:min_g_size] for g_idx in g_idxs])
    return dataset.subset(idx)


def process_dataset(dataset, group_balance=False, max_n=None, random_selection=False):
    """Get the original split.
    Returns: processed_dataset
    """
    if max_n is not None:
        if random_selection:
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            idx = idx[:max_n]
        else:
            idx = np.s_[:max_n]
        dataset = dataset.subset(idx)

    if group_balance:
        dataset = group_balanced(dataset)

    return dataset


def get_val_set(datasets, group_balance=False, add_train=True, random_selection=True):
    dataset = process_dataset(datasets["val"], group_balance=group_balance)
    if add_train:
        dataset = concatenate_datasets(
            process_dataset(
                datasets["train"],
                max_n=len(dataset), random_selection=random_selection),
            dataset)
    return dataset


def get_train_val_set(datasets, group_balance=False, max_n=None, random_selection=False):
    """Get original train and val sets.
    Returns: (train_dataset, val_dataset)
    """
    return (
        process_dataset(
            datasets["train"],
            group_balance=group_balance,
            max_n=max_n,
            random_selection=random_selection),
        process_dataset(
            datasets["val"],
            group_balance=False)
    )


def split_val_set(datasets, n_val=None, group_balance=False, add_train=True):
    """Split a valtrain set from the val set and optionally merge with part of the train set.
    Args:
        n_val: Number of remaining val examples. Default: len(val_set) // 2
        group_balance: Whether to make the valtrain set group balanced.
        add_train: Whether to add part of the train set.
    Returns: (train_set, val_set)
    """
    val_set = datasets["val"]

    if n_val is None:
        n_val = len(val_set) // 2

    # randomly split val set
    idx = np.arange(len(val_set))
    np.random.shuffle(idx)
    train_set = val_set.subset(idx[n_val:])
    val_set = val_set.subset(idx[:n_val])

    if group_balance:
        train_set = group_balanced(train_set)

    if add_train:
        train_set = concatenate_datasets(
            get_split(datasets["train"],
                      max_n=len(train_set), random_selection=False),
            train_set)

    return train_set, val_set


def build_logistic_regression_model(
        hypers, d=None, logreg_kwargs=dict(solver="liblinear")):
    return LogisticRegression(
        **hypers.to_kwargs(), **logreg_kwargs)


def get_conf_acc(logits, target):
    probs = sp.special.softmax(logits, axis=-1)
    pred_ = np.argmax(probs, axis=-1, keepdims=True)
    pred = pred_.squeeze(axis=-1)
    conf = np.take_along_axis(probs, pred_, axis=-1).squeeze(axis=-1)
    acc = pred == target
    return conf, acc


def get_ece(conf, acc, n_bins=10, conf_low=.5, verbose=True, ax: Optional[matplotlib.axes.Axes] = None):
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

    bin_low = int(conf_low * n_bins)
    overconf = sum_overconf / bin_counts
    if verbose:
        print(f"ECE={ece:.4f}")
        print("bin_counts=" + " ".join(map("{:6d}".format, bin_counts[bin_low:])))
        print("  overconf=" + " ".join(map("{:6.3f}".format, overconf[bin_low:])))
    if ax is not None:
        mean_acc = sum_acc / bin_counts
        width = 1 / n_bins
        bin_centers = np.linspace(0, 1 - width, n_bins) + .5 * width
        acc_bar = ax.bar(bin_centers[bin_low:], mean_acc[bin_low:], width=width, alpha=1.0, color="blue")
        overconf_bar = ax.bar(bin_centers[bin_low:], overconf[bin_low:], bottom=mean_acc[bin_low:], width=width, color="red", alpha=0.5, hatch='//', edgecolor='r')

        margin = conf * acc + (1 - conf) * (1 - acc)
        margin_bin_counts = np.ndarray((n_bins,), dtype=int)
        for i_bin in range(n_bins):
            a, b = i_bin / n_bins, (i_bin + 1) / n_bins
            if a == 0:
                a -= 1
            subsamples = (margin > a) & (margin <= b)
            margin_bin_counts[i_bin] = subsamples.sum()
        dist = margin_bin_counts / len(conf)
        dist_bar = ax.bar(bin_centers, dist * 0.2, width=width, color="magenta")

        ax.axline((0, 0), slope=1, linestyle="--", color="gray")
        ax.legend([acc_bar, overconf_bar, dist_bar], ["Outputs", "Gap", "Dist."], loc="best")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return ece


def evaluate_on_dataset(
        pred_probs, dataset, with_ece=False, n_bins=10,
        verbose=True, result_path=None):
    """Evaluate model on dataset.
    Args:
        pred_probs: predicted probabilities of shape (n, n_classes)
        dataset: dataset
        with_ece: bool, whether to compute ECE. If False, returned
            ece, group_eces will both be None
        n_bins: int, number of bins in computing ECE
        verbose: bool, whether to print evaluation information
    Returns:
        preds, corrects, group_accs, ece, group_eces
    """
    preds = pred_probs.argmax(axis=-1)
    corrects = preds == dataset.y
    group_accs = [corrects[dataset.g == g_id].mean() for g_id in range(dataset.n_groups)]

    if with_ece:
        conf = np.take_along_axis(pred_probs, np.expand_dims(preds, axis=-1), axis=-1).squeeze(axis=-1)  # confidence
        plotting = result_path is not None
        if plotting:  # plot
            n_rows, n_cols = dataset.n_classes, dataset.n_places
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False, sharex=True, sharey=True)
            group_axs = list(chain.from_iterable(axs))
        ece = get_ece(conf, corrects, n_bins=n_bins, verbose=verbose)
        group_eces = [
            get_ece(conf[dataset.g == g_id], corrects[dataset.g == g_id], n_bins=n_bins,
                    verbose=verbose, ax=(group_axs[g_id] if plotting else None))
            for g_id in range(dataset.n_groups)]
        if plotting:
            result_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(result_path/"group_calibration.pdf")
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
        hyper_options, get_datasets, scaler="train", num_retrains=1,
        build_model=build_logistic_regression_model,
        objective=worst_group_objective,
        with_ece=False,
        verbose=False):
    """Tune hyperparameters of the model to minimize the objective.
    Args:
        hyper_options: iterable of HyperParams, hyper-parameter options to try.
        get_datasets: callable to get train and val sets.
        scaler: If set to "train", fit the train set each time from get_datasets. None for no preprocessing.
        num_retrains: int, number of calls to get_datasets()
        build_model: Callable, model(hypers, d) builds the model with hypers
            and dimension d.
            model.fit(train_set.embedding, train_set.y) train the model, and
            model.predict_proba(val_set.embedding) returns predictive
            probabilities over classes on val_set.embedding.
        objective: Callable, objective(*eval_result) returns
            objective, info_str which are the objective value and the
            infomation string to print.
        with_ece: bool, whether to compute ECE.
        verbose: bool, whether to print evaluation information.
    """
    objectives = defaultdict(float)
    for i in range(num_retrains):
        train_set, val_set = get_datasets()
        print(f"train group sizes: {np.bincount(train_set.g)}")
        if scaler == "train":
            _scaler = StandardScaler()
            _scaler.fit(train_set.embedding)
        else:
            _scaler = scaler
        if _scaler is not None:
            train_set = train_set.transform_embedding(_scaler.transform)
            val_set = val_set.transform_embedding(_scaler.transform)

        for hypers in hyper_options:
            model = build_model(hypers, train_set.embedding.shape[-1])
            model.fit(train_set.embedding, train_set.y)
            val_pred_probs = model.predict_proba(val_set.embedding)
            eval_result = evaluate_on_dataset(
                val_pred_probs, val_set,
                with_ece=with_ece, verbose=verbose)
            obj, info_str = objective(*eval_result)
            objectives[hypers] += obj
            print(f"{hypers} {info_str}")

    ks, vs = list(objectives.keys()), list(objectives.values())
    best_hypers = ks[np.argmin(vs)]
    return best_hypers


def eval(model, datasets, verbose=True, result_path=None):
    results = {}
    for split, dataset in datasets.items():
        pred_probs = model.predict_proba(dataset.embedding)
        preds, corrects, group_accs, ece, group_eces = evaluate_on_dataset(
            pred_probs, dataset, with_ece=True, verbose=verbose,
            result_path=(result_path/split if result_path is not None else None))
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


def dfr_eval(
        hypers, get_train_dataset, get_eval_dataset, scaler,
        num_retrains=20,
        logreg_kwargs=dict(solver="liblinear"),
        verbose=True,
        result_path=None):
    coefs, intercepts = [], []

    for i in range(num_retrains):
        train_set = get_train_dataset()
        print(f"train group sizes: {np.bincount(train_set.g)}")
        if scaler is not None:
            train_set = train_set.transform_embedding(scaler.transform)

        logreg = build_logistic_regression_model(
            hypers, logreg_kwargs=logreg_kwargs)
        logreg.fit(train_set.embedding, train_set.y)

        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    test_set = get_eval_dataset()
    print(f"test group sizes: {np.bincount(test_set.g)}")
    if scaler is not None:
        test_set = test_set.transform_embedding(scaler.transform)

    logreg = build_logistic_regression_model(
        hypers, logreg_kwargs=logreg_kwargs)
    n_classes = train_set.n_classes
    # the fit is only needed to set up logreg
    logreg.fit(train_set.embedding[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    datasets = {
        "test": test_set,
        "train": train_set,
    }
    return eval(logreg, datasets, verbose=verbose, result_path=result_path)


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
    n_classes = datasets["train"].n_classes
    model = torchvision.models.resnet50(weights=None)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.cuda()
    model.eval()

    all_results = {}

    dataset_name = get_dataset_name(args.data_dir)

    # Extract embeddings
    get_embed = create_feature_extractor(model, return_nodes=["flatten"])
    for split, loader in loaders.items():
        dataset = datasets[split]
        embedding_path = args.ckpt_path.parent/split/"embedding.npy"
        if embedding_path.exists():
            print(f"loading from {embedding_path}")
            dataset.embedding = np.load(embedding_path, allow_pickle=True)
            dataset.torch_embedding = torch.from_numpy(dataset.embedding).cuda()
        else:
            embedding = []
            for x, y, g, p in tqdm.tqdm(loader):
                with torch.no_grad():
                    embedding.append(get_embed(x.cuda())["flatten"].detach())
            embedding = torch.cat(embedding).detach()
            dataset.torch_embedding = embedding
            dataset.embedding = embedding.cpu().numpy()
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"saving to {embedding_path}")
            np.save(embedding_path, dataset.embedding)

    scaler = StandardScaler()
    scaler.fit(datasets["train"].embedding)

    # Hyperparams options
    penalty_options = [args.penalty] if args.penalty != "tune" else PENALTY_OPTIONS
    options = DATASET_OPTIONS[dataset_name]

    result_path = args.result_path.parent
    result_path.mkdir(parents=True, exist_ok=True)

    for expr in args.expr:
        if expr == "base":  # Evaluate base model
            expr_desc = "Base Model"
            print(expr_desc)
            results_name = "base_model_results"

            if args.original_base_eval:
                get_yp_func = partial(get_y_p, n_places=datasets["train"].n_places)
                results = {}
                for split, loader in loaders.items():
                    split_results, logits = evaluate(model, loader, get_yp_func, return_logits=True)
                    dataset = loader.dataset
                    conf, acc = get_conf_acc(logits, dataset.y)
                    mean_accuracy = acc.mean()
                    np.testing.assert_approx_equal(mean_accuracy, split_results["mean_accuracy"])

                    base_result_path = result_path/expr/split
                    plotting = base_result_path is not None
                    if plotting:  # plot
                        n_rows, n_cols = dataset.n_classes, dataset.n_places
                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False, sharex=True, sharey=True)
                        group_axs = list(chain.from_iterable(axs))
                    ece = get_ece(conf, acc)
                    group_eces = [
                        get_ece(conf[dataset.g == g_id], acc[dataset.g == g_id],
                                ax=(group_axs[g_id] if plotting else None))
                        for g_id in range(dataset.n_groups)]
                    if plotting:
                        base_result_path.mkdir(parents=True, exist_ok=True)
                        plt.savefig(base_result_path/"group_calibration.pdf")

                    split_results["calibration"] = {
                        "ece": ece,
                        "group_eces": group_eces,
                    }
                    results[split] = split_result
                model.eval()

            else:
                classifier = SoftmaxClassifier(model.fc)
                for dataset in datasets.values():
                    dataset.numpy_embedding = dataset.embedding
                    dataset.embedding = dataset.torch_embedding
                results = eval(classifier, datasets, result_path=result_path/expr)
                for dataset in datasets.values():
                    dataset.embedding = dataset.numpy_embedding
                    del dataset.numpy_embedding

        elif expr == "on_val":  # DFR on validation
            expr_desc = "DFR on validation"
            print(expr_desc)
            results_name = "dfr_val_results"
            results = {}
            learn_class_weights = not(args.balance_dfr_val and args.notrain_dfr_val)
            class_weight_options = (
                options["class_weight"] if learn_class_weights else
                DEFAULT_CLASS_WEIGHT)
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, options["C"],
                        options["intercept_scaling"],
                        class_weight_options),
            )
            hyper = tune(
                hyper_options,
                partial(
                    split_val_set, datasets,
                    group_balance=args.balance_dfr_val,
                    add_train=not args.notrain_dfr_val
                ),
                build_model=build_logistic_regression_model
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(dfr_eval(
                hyper,
                partial(
                    get_val_set,
                    datasets,
                    group_balance=args.balance_dfr_val,
                    add_train=not args.notrain_dfr_val, random_selection=True
                ),
                partial(process_dataset, datasets["test"]),
                scaler,
                result_path=result_path/expr,
            ))

        elif expr == "on_train":  # DFR on train subsampled
            expr_desc = "DFR on train subsampled"
            print(expr_desc)
            results_name = "dfr_train_results"
            results = {}
            learn_class_weights = args.tune_class_weights_dfr_train
            class_weight_options = (
                options["class_weight"] if learn_class_weights else
                DEFAULT_CLASS_WEIGHT)
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, options["C"],
                        options["intercept_scaling"],
                        class_weight_options),
            )
            hyper = tune(
                hyper_options,
                partial(
                    get_train_val_set,
                    datasets,
                    group_balance=True
                ),
                scaler=scaler,
                build_model=partial(
                    build_logistic_regression_model,
                    logreg_kwargs=dict(solver="liblinear", max_iter=20)
                )
            )
            results["best_hypers"] = hyper
            print("Hypers:", hyper)
            results.update(dfr_eval(
                hyper,
                partial(process_dataset, datasets["train"], group_balance=True),
                partial(process_dataset, datasets["test"]),
                scaler,
                result_path=result_path/expr,
            ))

        elif expr == "on_unbalanced_train":  # DFR on unbalanced subsampled train
            n_train = len(datasets["train"])
            max_n = int(n_train * args.train_frac)
            expr_desc = f"DFR on unbalanced train ({args.train_frac:.2%}={max_n}/{n_train})"
            print(expr_desc)
            results_name = "dfr_unbalanced_train_results"
            results = {}
            hyper_options = map(
                HyperParams._make,
                product(penalty_options, options["C"],
                        options["intercept_scaling"] if False else [1.],  # reduce time
                        options["class_weight"]),
            )
            hyper = tune(
                hyper_options,
                partial(
                    get_train_val_set,
                    datasets,
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                scaler=scaler,
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
                    process_dataset, datasets["train"],
                    group_balance=False,
                    max_n=max_n,
                    random_selection=True
                ),
                partial(process_dataset, datasets["test"]),
                scaler,
                result_path=result_path/expr,
            ))

        print(expr_desc+" results:")
        print(json.dumps(results, indent=INDENT))
        print()
        all_results[results_name] = results

        with open(args.result_path, 'w') as f:
            json.dump(all_results, f, indent=INDENT)
