import pickle
import numpy as np
from sklearn import clone
from tqdm import tqdm
from modelcomp import constants
import warnings
from modelcomp.constants import METRICS
import os
import pandas as pd
import pickle as pkl
from abc import abstractmethod

__all__ = ["ModelComparison"]


class BaseMetric:
    def __init__(self, name, feature_names=None):
        self.__name = name
        self.__values = []
        self.__feature_names = feature_names

    def append(self, value):
        self.__values.append(value)

    def export(self, path, *measures, plots=False):
        os.makedirs(os.path.join(path, f"{self.name}"), exist_ok=True)
        for index, value in enumerate(self.__values):
            np.savetxt(
                os.path.join(path, f"{self.name}", f"split_{index}.csv"),
                value,
                delimiter=",",
            )
        if plots:
            self.plot().savefig(os.path.join(path, f"{self.name}.png"))

        for measure in measures:
            if measure == "mean":
                np.savetxt(
                    os.path.join(path, self.name, "mean.csv"),
                    [self.mean],
                    delimiter=",",
                )
            elif measure == "std":
                np.savetxt(
                    os.path.join(path, self.name, "std.csv"),
                    [self.std],
                    delimiter=",",
                )

    @abstractmethod
    def plot(self):
        pass
        

    @property
    @abstractmethod
    def mean(self):
        pass
        

    @property
    @abstractmethod
    def std(self):
        pass

    @property
    def name(self):
        return self.__name

    @property
    def values(self):
        return self.__values

    @property
    def feature_names(self):
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        self.__feature_names = feature_names

    def __getitem__(self, index):
        return self.__values[index]

    def __eq__(self, other):
        if not isinstance(other, BaseMetric):
            return False
        return (
            self.__name == other.__name
            and self.__values == other.__values
        )


class ScoreMetric(BaseMetric):
    def invoke(self, model, X, y, train, test):
        self.append(METRICS[self.name](y[test], model.predict(X[test])))

    @property
    def mean(self):
        return np.mean(self.values)

    @property
    def std(self):
        return np.std(self.values)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.values, marker="o", linestyle="-", color="b")
        ax.set_title(f"{self.name} over splits")
        ax.set_xlabel("Split")
        ax.set_ylabel(self.name)
        ax.grid(True)
        return fig


class CurveMetric(BaseMetric):
    def __init__(self, name, feature_names=None):
        super().__init__(name, feature_names)
        self.y_true = []
        self.y_pred = []

    def invoke(self, model, X, y, train, test):
        y_pred = model.predict_proba(X[test])[:, 1]
        if self.name == "precision_recall_curve":
            precision, recall, thresholds = METRICS[self.name](y[test], y_pred)
            self.y_true.append(np.array(y[test]))
            self.y_pred.append(np.array(y_pred))
            self.append(np.vstack([precision, recall, np.append(thresholds, np.nan)]))
        elif self.name == "roc_curve":
            fpr, tpr, thresholds = METRICS[self.name](y[test], y_pred)
            self.append(np.vstack([fpr, tpr, thresholds]))

    @property
    def mean(self):
        if self.name == "roc_curve":
            interp_values = []
            for values in self.values:
                mean_tpr = np.interp(np.linspace(0, 1, 101), values[0], values[1])
                mean_tpr[0] = 0.0
                mean_tpr[-1] = 1.0
                interp_values.append(mean_tpr)
            return np.linspace(0, 1, 101), np.mean(interp_values, axis=0)
        elif self.name == "precision_recall_curve":
            y_true = np.concatenate(self.y_true)
            y_pred = np.concatenate(self.y_pred)
            precision, recall, _ = METRICS[self.name](y_true, y_pred)
            return precision, recall

    @property
    def std(self):
        if self.name == "roc_curve":
            interp_values = []
            for values in self.values:
                mean_tpr = np.interp(np.linspace(0, 1, 101), values[0], values[1])
                mean_tpr[0] = 0.0
                mean_tpr[-1] = 1.0
                interp_values.append(mean_tpr)
            return np.linspace(0, 1, 101), np.std(interp_values, axis=0)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if self.name == "roc_curve":
            for values in self.values:
                ax.plot(values[0], values[1], linestyle="-", alpha=0.3)
            mean_fpr, mean_tpr = self.mean
            ax.plot(mean_fpr, mean_tpr, color="b", label="Mean ROC curve")
            tprs_upper = np.minimum(mean_tpr + self.std, 1)
            tprs_lower = np.maximum(mean_tpr - self.std, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.3,
                label="±1 std. dev.",
            )
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(True)
        elif self.name == "precision_recall_curve":
            for values in self.values:
                ax.plot(values[1], values[0], linestyle="-", alpha=0.3)
            mean_precision, mean_recall = self.mean
            ax.plot(
                mean_recall,
                mean_precision,
                color="b",
                label="Mean Precision-Recall curve",
            )
            precisions_upper = np.minimum(mean_precision + self.std, 1)
            precisions_lower = np.maximum(mean_precision - self.std, 0)
            ax.fill_between(
                mean_recall,
                precisions_lower,
                precisions_upper,
                color="grey",
                alpha=0.3,
                label="±1 std. dev.",
            )
            ax.set_title("Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            ax.grid(True)
        return fig


class ImportanceMetric(BaseMetric):
    def invoke(self, model, X, y, train, test):
        self.__X = X
        if self.name == "builtin_importance":
            self.append(constants.builtin_importance(model))
        elif self.name == "shap_values":
            self.append(constants.shap_values(model, test, X[test]))

    @property
    def mean(self):
        return np.mean(np.abs(self.values), axis=0)

    @property
    def std(self):
        return np.std(np.abs(self.values), axis=0)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if self.name == "builtin_importance":
            feature_importance = self.mean
            if self.feature_names is not None:
                ax.bar(self.feature_names, feature_importance)
            else:
                ax.bar(
                    [f"f{i}" for i in range(len(feature_importance))],
                    feature_importance,
                )
            ax.set_title("Feature Importance")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Importance")
            ax.grid(True)
        elif self.name == "shap_values":
            import shap
            shap_values = self.values
            if self.feature_names is not None:
                shap.summary_plot(shap_values, self.__X, feature_names=self.feature_names)
            else:
                shap.summary_plot(shap_values, self.__X)
        return fig


class MetricFactory:
    @staticmethod
    def create_metric(name, feature_names=None):
        if name in ["accuracy_score", "precision_score", "recall_score", "f1_score"]:
            return ScoreMetric(name)
        elif name in ["precision_recall_curve", "roc_curve"]:
            return CurveMetric(name)
        elif name in ["builtin_importance", "shap_values"]:
            return ImportanceMetric(name, feature_names)
        else:
            warnings.warn(f"Metric {name} not supported")
            return None
class ModelStats:
    def __init__(self, model, *metrics, feature_names=None):
        self.__model = model
        self.__fit_models = []
        for metric in metrics:
            metric_to_set = MetricFactory.create_metric(name=metric, feature_names=feature_names)
            if metric_to_set is not None:
                setattr(self, metric, metric_to_set)

    def append_metrics(self, **metrics):
        for metric in metrics.keys():
            if metric not in METRICS:
                raise ValueError(f"{metric} not found in METRICS")
            if not hasattr(self, metric):
                setattr(self, metric, MetricFactory.create_metric(name=metric))

            if metrics[metric] is not None:
                getattr(self, metric).append(metrics[metric])

    def export(self, path, *measures, plots=False):
        for metric in self:
            metric.export(path, *measures, plots=plots)

    def __iter__(self):
        return (
            metric for metric in self.__dict__.values() if isinstance(metric, BaseMetric)
        )

    def __len__(self):
        return len(
            [metric for metric in self.__dict__.values() if isinstance(metric, BaseMetric)]
        )

    @property
    def model(self):
        return self.__model

    @property
    def fit_models(self):
        return self.__fit_models

    def append_fit_model(self, fit_model):
        self.__fit_models.append(fit_model)

    def __eq__(self, other):
        if not isinstance(other, ModelStats):
            return False
        return (
            self.__model == other.__model
            and self.__fit_models == other.__fit_models
            and all(
                getattr(self, metric) == getattr(other, metric) for metric in METRICS
            )
        )


class ModelComparison:

    def __init__(self, models, validation, X, y, feature_names=None):
        self.__model_stats = [
            ModelStats(model, feature_names=feature_names) for model in models
        ]
        self.__splits = []
        self.__validation = validation
        self.__feature_names = feature_names
        if X is not None and y is not None:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
            if type(X) is not np.ndarray:
                X = np.array(X)
            if type(y) is not np.ndarray:
                y = np.array(y)
            self.__X = X
            self.__y = y

    @staticmethod
    def from_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found")
        elif not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        elif not os.access(path, os.R_OK):
            raise PermissionError(f"Permission denied to read from {path}")
        else:
            modelcomparison = ModelComparison(models=[], validation=None, X=None, y=None)
            modelcomparison.__splits = []
            for idx in range(len(os.listdir(os.path.join(path, "splits")))):
                general_split_dir = os.path.join(path, "splits", f"split_{idx}")
                train_indices_path = os.path.join(general_split_dir, "train.csv")
                test_indices_path = os.path.join(general_split_dir, "test.csv")

                train = np.loadtxt(train_indices_path, delimiter=",", dtype=int)
                test = np.loadtxt(test_indices_path, delimiter=",", dtype=int)
                modelcomparison.__splits.append((train, test))

            model_stats = []
            for model_idx in range(len(os.listdir(path)) - 1):
                model_dir = os.path.join(path, f"model_{model_idx}")
                model_pkl_path = os.path.join(model_dir, "model.pkl")

                with open(model_pkl_path, "rb") as model_pkl_file:
                    model = clone(pickle.load(model_pkl_file))

                model_stats.append(ModelStats(model))

                if os.path.exists(os.path.join(model_dir, "fit_models")):
                    fit_models_dir = os.path.join(model_dir, "fit_models")
                    for split_index in range(len(modelcomparison.__splits)):
                        fit_model_pkl_path = os.path.join(
                            fit_models_dir, f"split_{split_index}.pkl"
                        )
                        with open(fit_model_pkl_path, "rb") as fit_model_pkl_file:
                            fit_model = pickle.load(fit_model_pkl_file)
                        model_stats[model_idx].append_fit_model(fit_model)

                for metric in METRICS.keys():
                    if os.path.exists(os.path.join(model_dir, metric)):
                        metric_dir = os.path.join(model_dir, metric)
                        metric_instance = MetricFactory.create_metric(name=metric)
                        for split_index in range(len(modelcomparison.__splits)):
                            metric_csv_path = os.path.join(
                                metric_dir, f"split_{split_index}.csv"
                            )
                            metric_values = np.loadtxt(metric_csv_path, delimiter=",")
                            metric_instance.append(metric_values)
                            setattr(model_stats[model_idx], metric, metric_instance)
                if os.path.exists(os.path.join(model_dir, "y_true")):
                    for split_index in range(len(modelcomparison.__splits)):
                        y_true_csv_path = os.path.join(
                            model_dir, "y_true", f"split_{split_index}.csv"
                        )
                        y_true_values = np.loadtxt(y_true_csv_path, delimiter=",")
                        model_stats[model_idx].precision_recall_curve.y_true.append(
                            y_true_values
                        )
                        y_pred_csv_path = os.path.join(
                            model_dir, "y_pred", f"split_{split_index}.csv"
                        )
                        y_pred_values = np.loadtxt(y_pred_csv_path, delimiter=",")
                        model_stats[model_idx].precision_recall_curve.y_pred.append(
                            y_pred_values
                        )
            modelcomparison.__model_stats = model_stats
            return modelcomparison

    def validate(
        self, *pipeline_metrics, keep_models=False, models=None, new_splits=False
    ):
        if new_splits:
            if self.__splits != []:
                warnings.warn(
                    "Train and test sets already exist and will be overwritten with new ones."
                )
                self.__splits = []
            for train, test in self.__validation.split(self.__X, self.__y):
                self.__splits.append((train, test))
        else:
            if self.__splits == []:
                warnings.warn(
                    "Train and test sets do not exist and new_splits=False. Creating new splits anyway."
                )
                self.__splits = []
                for train, test in self.__validation.split(self.__X, self.__y):
                    self.__splits.append((train, test))

        for model_idx in range(len(self.__model_stats)):

            if any(
                isinstance(attr, BaseMetric)
                for attr in self.__model_stats[model_idx].__dict__.values()
            ):
                warnings.warn(
                    f"Metrics for model {model_idx} already exist and will be overwritten."
                )

            self.__model_stats[model_idx].append_metrics(
                **{pipeline_metric: None for pipeline_metric in pipeline_metrics}
            )

            for train, test in tqdm(self.__splits, desc="Split"):
                fit_model = clone(self.__model_stats[model_idx].model)
                fit_model.fit(self.__X[train], self.__y[train])
                if keep_models:
                    self.__model_stats[model_idx].append_fit_model(fit_model)
                # fitting model

                for metric in self.__model_stats[model_idx]:
                    metric.feature_names = self.__feature_names
                    metric.invoke(fit_model, self.__X, self.__y, train, test)

    def remove_model(self, model):
        self.models.remove(model)
        indices_to_remove = [
            i
            for i in range(len(self.__model_stats))
            if self.__model_stats[i].model == model
        ]
        for i in sorted(indices_to_remove, reverse=True):
            del self.__model_stats[i]

    def export(self, path, *measures, plots=False):
        for idx, (train, test) in enumerate(self.__splits):
            general_split_dir = os.path.join(path, "splits", f"split_{idx}")
            os.makedirs(general_split_dir, exist_ok=True)
            train_indices_path = os.path.join(general_split_dir, "train.csv")
            test_indices_path = os.path.join(general_split_dir, "test.csv")

            np.savetxt(train_indices_path, train, delimiter=",", fmt="%d")
            np.savetxt(test_indices_path, test, delimiter=",", fmt="%d")

        for model_idx in range(len(self.__model_stats)):
            model = clone(self.__model_stats[model_idx].model)
            model_stats = self.__model_stats[model_idx]
            model_dir = os.path.join(path, f"model_{model_idx}")
            if os.path.exists(model_dir):
                warnings.warn(
                    f"CSV files for model {model} already exist in {model_dir}. Overwriting."
                )

            os.makedirs(model_dir, exist_ok=True)

            np.savetxt(
                os.path.join(model_dir, "name.txt"),
                [str(model)],
                delimiter=",",
                fmt="%s",
            )

            pkl.dump(model, open(os.path.join(model_dir, "model.pkl"), "wb"))

            for split_index, (train, test) in enumerate(self.__splits):

                if model_stats.fit_models:
                    model_pkl_path = os.path.join(
                        model_dir, "fit_models", f"split_{split_index}.pkl"
                    )
                    with open(model_pkl_path, "wb") as model_pkl_file:
                        pickle.dump(model_stats.fit_models[split_index], model_pkl_file)

                model_stats.export(model_dir, *measures, plots=plots)
        if plots:
            for metric in METRICS:
                fig = self.plot(metric)
                fig.savefig(os.path.join(path, f"{metric}.png"))

    def plot(self, metric_name):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        fig, ax = plt.subplots()
        for model_idx, model_stat in enumerate(self.__model_stats):
            metric = getattr(model_stat, metric_name, None)
            if metric is None:
                raise ValueError(
                    f"Metric {metric_name} not found for model {model_idx}"
                )
            if isinstance(metric, ScoreMetric):
                ax.plot(
                    metric.values, marker="o", linestyle="-", label=f"Model {model_idx}"
                )
            elif isinstance(metric, CurveMetric):
                if metric_name == "roc_curve":
                    mean_fpr, mean_tpr = metric.mean
                    ax.plot(
                        mean_fpr, mean_tpr, label=f"Model {model_idx} Mean ROC curve"
                    )
                elif metric_name == "precision_recall_curve":
                    mean_precision, mean_recall = metric.mean
                    ax.plot(
                        mean_recall,
                        mean_precision,
                        label=f"Model {model_idx} Mean Precision-Recall curve",
                    )
                else:
                    raise ValueError(
                        f"Plotting not supported for metric type {type(metric)}"
                    )
            else:
                raise ValueError(
                    f"Plotting not supported for metric type {type(metric)}"
                )

        ax.set_title(f"Comparison of {metric_name}")
        ax.set_xlabel(
            "Split"
            if isinstance(metric, ScoreMetric)
            else (
                "Recall"
                if metric_name == "precision_recall_curve"
                else "False Positive Rate"
            )
        )
        ax.set_ylabel(
            metric_name
            if isinstance(metric, ScoreMetric)
            else (
                "Precision"
                if metric_name == "precision_recall_curve"
                else "True Positive Rate"
            )
        )
        ax.legend(loc="best")
        ax.grid(True)
        return fig

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @X.setter
    def X(self, X):
        self.__X = X

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def splits(self):
        return self.__splits

    @splits.setter
    def splits(self, splits):
        self.__splits = splits

    @property
    def feature_names(self):
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        self.__feature_names = feature_names

    @property
    def validation(self):
        return self.__validation

    @validation.setter
    def validation(self, validation):
        self.__validation = validation

    @property
    def models(self):
        return self.__model_stats.keys()

    @models.setter
    def models(self, models):
        for model in models:
            if model not in self.__model_stats:
                self.__model_stats[model] = None
        for model in self.__model_stats:
            if model not in models:
                del self.__model_stats[model]

    def add_model(self, model):
        self.__model_stats[model] = None

    def remove_model(self, model):
        del self.__model_stats[model]

    def __eq__(self, other):
        if not isinstance(other, ModelComparison):
            return False
        return (
            self.__model_stats == other.__model_stats
            and self.__splits == other.__splits
            and self.__validation == other.__validation
            and np.array_equal(self.__X, other.__X)
            and np.array_equal(self.__y, other.__y)
            and self.__feature_names == other.__feature_names
        )
