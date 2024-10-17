import pickle
import numpy as np
from sklearn import clone
from modelcomp import _constants
import warnings
import modelcomp._constants as _constants
import os.path
import pickle as pkl
from abc import abstractmethod
import modelcomp._sysutils as _sysutils

__all__ = ["ModelComparison"]


class BaseMetric:
    def __init__(self, name, model_stats):
        self._model_stats = model_stats
        self._name = name
        self._values = []

    def append(self, value):
        self._values.append(value)

    def export(self, path, *measures, plots=False):
        _sysutils.check_dir_write(path, force_create=True)
        for index, value in enumerate(self._values):
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            if isinstance(value, (np.ndarray, list)):
                np.savetxt(
                    os.path.join(path, f"split_{index}.csv"),
                    value,
                    delimiter=",",
                )
            else:
                np.savetxt(
                    os.path.join(path, f"split_{index}.csv"),
                    [value],
                    delimiter=",",
                )
        if plots:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError("matplotlib is required for plotting")
            if isinstance(self, CurveMetric):
                self.plot().savefig(
                    os.path.join(path, f"{self.name}.png")
                )
            else:
                self.plot().savefig(os.path.join(path, f"{self.name}.png"))
            plt.close()

        for measure in measures:
            if measure == "mean":
                if self.mean is not None:
                    if isinstance(value, np.ndarray) and value.shape == ():
                        value = value.item()
                    if isinstance(value, (np.ndarray, list)):
                        np.savetxt(
                            os.path.join(path, "mean.csv"),
                            self.mean,
                            delimiter=",",
                        )
                    else:
                        np.savetxt(
                            os.path.join(path, "mean.csv"),
                            [self.mean],
                            delimiter=",",
                        )
                else:
                    warnings.warn(f"No mean for metric {self.name}")
            elif measure == "std":
                if self.std is not None:
                    if isinstance(value, np.ndarray) and value.shape == ():
                        value = value.item()
                    if isinstance(value, (np.ndarray, list)):
                        np.savetxt(
                            os.path.join(path, "std.csv"),
                            self.std,
                            delimiter=",",
                        )
                    else:
                        np.savetxt(
                            os.path.join(path, "std.csv"),
                            [self.std],
                            delimiter=",",
                        )
                else:
                    warnings.warn(f"No standard deviation for metric {self.name}")
            else:
                warnings.warn(f"Unrecognized measure {measure}")

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
        return self._name

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __eq__(self, other):
        if not isinstance(other, BaseMetric):
            return False
        return self._name == other._name and self._values == other._values


class ScoreMetric(BaseMetric):
    def invoke(self, model, X, y, train, test):
        self.append(_constants.METRICS[self.name](y[test], model.predict(X[test])))

    @staticmethod
    def from_path(path, model_stats):
        _sysutils.check_dir_read(path)

        metric = ScoreMetric(name=os.path.basename(path), model_stats=model_stats)
        i = 0
        while _sysutils.check_file_read(
            os.path.join(path, f"split_{i}.csv"), raise_errors=False
        ):
            values = np.loadtxt(os.path.join(path, f"split_{i}.csv"), delimiter=",")
            metric.append(values)
            i += 1
        return metric

    @property
    def mean(self):
        return np.mean(self.values)

    @property
    def std(self):
        return np.std(self.values)

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter([""] * len(self.values), self.values)
        ax.set_xlabel("")
        ax.set_title(f"{self.name} scatter plot")
        ax.set_ylabel(self.name)
        return fig


class CurveMetric(BaseMetric):
    def invoke(self, model, X, y, train, test):
        y_pred_proba = model.predict_proba(X[test])[:, 1]
        if self.name == "roc_curve":
            fpr, tpr, thresholds = _constants.METRICS[self.name](y[test], y_pred_proba)
            self.append(np.vstack([fpr, tpr, thresholds]))
        if self.name == "precision_recall_curve":
            precision, recall, thresholds = _constants.METRICS[self.name](
                y[test], y_pred_proba
            )
            self._model_stats.y_true.append(np.array(y[test]))
            self._model_stats.y_pred.append(np.array(y_pred_proba))
            self.append(np.vstack([precision, recall, np.append(thresholds, np.nan)]))

    @staticmethod
    def from_path(path, model_stats):
        _sysutils.check_dir_read(path)
        metric = CurveMetric(name=os.path.basename(path), model_stats=model_stats)
        i = 0
        while _sysutils.check_file_read(
            os.path.join(path, f"split_{i}.csv"), raise_errors=False
        ):
            values = np.loadtxt(os.path.join(path, f"split_{i}.csv"), delimiter=",")
            metric.append(values)
            i += 1
        return metric

    @property
    def mean(self):
        return _constants.METRICS[self.name](
            np.concatenate(self._model_stats.y_true), np.concatenate(self._model_stats.y_pred)
        )[0:2]

    @property
    def std(self):
        pass

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if self.name == "roc_curve":
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                raise ImportError("sklearn is required for plotting ROC curves")
            aucs = []
            for values_idx, values in enumerate(self.values):
                ax.plot(values[0], values[1], linestyle="-", alpha=0.3)
                aucs.append(
                    roc_auc_score(self._model_stats.y_true[values_idx], self._model_stats.y_pred[values_idx])
                )
            mean_fpr, mean_tpr = self.mean
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC Curve (AUC = %0.2f $\pm$ %0.2f)"
                % (
                    roc_auc_score(
                        np.concatenate(self._model_stats.y_true), np.concatenate(self._model_stats.y_pred)
                    ),
                    np.std(aucs),
                ),
                lw=2,
                alpha=0.8,
            )
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(True)
        elif self.name == "precision_recall_curve":
            try:
                from sklearn.metrics import average_precision_score
            except ImportError:
                raise ImportError(
                    "sklearn is required for plotting Precision-Recall curves"
                )
            pr_aucs = []
            for split_idx, values in enumerate(self.values):
                ax.plot(values[1], values[0], linestyle="-", alpha=0.3)
                pr_aucs.append(
                    average_precision_score(
                        self._model_stats.y_true[split_idx], self._model_stats.y_pred[split_idx]
                    )
                )
            mean_precision, mean_recall = self.mean
            ax.plot(
                mean_recall,
                mean_precision,
                color="b",
                label=r"Mean PR Curve (AUC = %0.2f $\pm$ %0.2f)"
                % (
                    average_precision_score(
                        np.concatenate(self._model_stats.y_true), np.concatenate(self._model_stats.y_pred)
                    ),
                    np.std(pr_aucs),
                ),
                lw=2,
                alpha=0.8,
            )
            ax.set_title("Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            ax.grid(True)
        return fig


class ImportanceMetric(BaseMetric):
    def __init__(self, name, model_stats, feature_names=None):
        super().__init__(name, model_stats)
        self.__test_splits = []
        if self.name == "builtin_importance":
            pass
        elif self.name == "shap_explainer":
            self.__explainer = None
        else:
            pass

    @staticmethod
    def from_path(path, model_stats):
        _sysutils.check_dir_read(path)
        metric = ImportanceMetric(name=os.path.basename(path), model_stats=model_stats)
        if metric.name == "builtin_importance":
            i = 0
            while _sysutils.check_file_read(
                os.path.join(path, f"split_{i}.csv"), raise_errors=False
            ):
                values = np.loadtxt(os.path.join(path, f"split_{i}.csv"), delimiter=",")
                metric.append(values)
                i += 1
        if metric.name == "shap_explainer":
            metric.__explainer = pkl.load(
                open(os.path.join(path, f"{metric.name}.pkl"), "rb")
            )
        return metric

    def invoke(self, model, X, y, train, test):
        self.__X = X
        self.__test_splits.append(test)
        if self.name == "builtin_importance":
            self.append(_constants.builtin_importance(model))
        elif self.name == "shap_explainer":
            self.shap_append(_constants.shap_explainer(model, test, X[test]))

    def export(self, path, *measures, plots=False):
        if self.name == "builtin_importance":
            super().export(path, *measures, plots=plots)
        elif self.name == "shap_explainer":
            if self.__explainer is not None:
                _sysutils.check_dir_write(path, force_create=True)
                pkl.dump(
                    self.__explainer, open(os.path.join(path, f"{self.name}.pkl"), "wb")
                )
            if plots:
                self.plot().savefig(os.path.join(path, f"{self.name}.png"))
        else:
            pass

    def shap_append(self, value):
        try:
            import shap
        except ImportError:
            raise ImportError(f"shap is required for SHAP values calculation")
        if type(value) is not shap.Explanation:
            raise ValueError(f"SHAP values must be of type shap.Explanation")

        value.values = value.values[:, :, 1]

        if self.__explainer == None:
            self.__explainer = value
        else:

            self.__explainer = shap.Explanation(
                data=np.append(self.__explainer.data, value.data, axis=0),
                base_values=np.append(self.__explainer.base_values, value.base_values),
                values=np.append(self.__explainer.values, value.values, axis=0),
                feature_names=self._model_stats.feature_names,
            )

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if self.name == "builtin_importance":
            feature_importance = self.mean
            # Get the indices of the 20 highest values
            top_indices = np.argsort(feature_importance)[-20:]
            # Plot the 20 highest values
            if self._model_stats.feature_names is not None:
                ax.barh(
                    [self._model_stats.feature_names[i] for i in top_indices],
                    feature_importance[top_indices],
                )
            else:
                ax.barh([f"f{i}" for i in top_indices], feature_importance[top_indices])
            ax.set_title("Feature Importance")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Importance")
            ax.grid(True)
        elif self.name == "shap_explainer":
            try:
                import shap
            except ImportError:
                raise ImportError(f"shap is required for SHAP values calculation")
            shap.plots.beeswarm(self.__explainer, max_display=20, show=False)
            fig = plt.gcf()
        return fig

    @property
    def mean(self):
        if self.name == "builtin_importance":
            return np.mean(np.abs(self.values), axis=0)
        elif self.name == "shap_explainer":
            return np.mean(np.abs(self.__explainer.values), axis=0)

    @property
    def std(self):
        if self.name == "builtin_importance":
            return np.std(np.abs(self.values), axis=0)
        elif self.name == "shap_explainer":
            return np.std(np.abs(self.__explainer.values), axis=0)


class MetricFactory:
    @staticmethod
    def create_metric(name, model_stats, feature_names=None):
        if name in ["accuracy_score", "precision_score", "recall_score", "f1_score"]:
            return ScoreMetric(name, model_stats)
        elif name in ["precision_recall_curve", "roc_curve"]:
            return CurveMetric(name, model_stats)
        elif name in ["builtin_importance", "shap_explainer"]:
            return ImportanceMetric(name, model_stats, feature_names)
        else:
            warnings.warn(f"Metric {name} not supported")
            return None


class ModelStats:
    def __init__(self, model, *metrics, feature_names=None):
        self.__model = model
        self.__fit_models = []
        self.__y_true = []
        self.__y_pred = []
        self.__feature_names = feature_names
        for metric in metrics:
            metric_to_set = MetricFactory.create_metric(
                name=metric, feature_names=feature_names
            )
            if metric_to_set is not None:
                setattr(self, metric, metric_to_set)

    def invoke(self, fit_model, X, y, train, test):
        self.__y_true.append(y[test])
        self.__y_pred.append(fit_model.predict_proba(X[test])[:, 1])
        for metric in self:
            if isinstance(metric, CurveMetric):
                metric.invoke(
                    fit_model, X, y, train, test,
                )
            else:
                metric.invoke(fit_model, X, y, train, test)
    @staticmethod
    def from_path(path):
        _sysutils.check_dir_read(path)
        model_stats = ModelStats(model=None)
        model_stats.__model = pkl.load(open(os.path.join(path, "model.pkl"), "rb"))
        if _sysutils.check_dir_read(
            os.path.join(path, "fit_models"), raise_errors=False
        ):
            for split_index in range(len(os.listdir(os.path.join(path, "fit_models")))):
                with open(
                    os.path.join(path, "fit_models", f"split_{split_index}.pkl"), "rb"
                ) as model_pkl_file:
                    model_stats.__fit_models.append(pickle.load(model_pkl_file))
        if _sysutils.check_dir_read(os.path.join(path, "y_true"), raise_errors=False):
            for split_index in range(len(os.listdir(os.path.join(path, "y_true")))):
                model_stats.__y_true.append(
                    np.loadtxt(
                        os.path.join(path, "y_true", f"split_{split_index}.csv"),
                        delimiter=",",
                    )
                )
        if _sysutils.check_dir_read(os.path.join(path, "y_pred"), raise_errors=False):
            for split_index in range(len(os.listdir(os.path.join(path, "y_pred")))):
                model_stats.__y_pred.append(
                    np.loadtxt(
                        os.path.join(path, "y_pred", f"split_{split_index}.csv"),
                        delimiter=",",
                    )
                )
        for metric in _constants.METRICS.keys():
            metric_path = os.path.join(path, metric)
            if os.path.exists(metric_path):
                if metric in _constants.CURVE_METRICS:
                    metric_type = CurveMetric
                elif metric in _constants.SCORE_METRICS:
                    metric_type = ScoreMetric
                elif metric in _constants.IMPORTANCE_METRICS:
                    metric_type = ImportanceMetric
                else:
                    warnings.warn(f"Metric {metric} not supported")
                    continue
                metric_instance = metric_type.from_path(metric_path, model_stats)
                setattr(model_stats, metric, metric_instance)
        return model_stats

    def append_metrics(self, **metrics):
        for metric in metrics.keys():
            if metric not in _constants.METRICS:
                raise ValueError(f"{metric} not found in _constants.METRICS")
            if not hasattr(self, metric):
                setattr(self, metric, MetricFactory.create_metric(name=metric, model_stats=self))

            if metrics[metric] is not None:
                getattr(self, metric).append(metrics[metric])

    def export(self, path, *measures, plots=False):
        if self.__fit_models != []:
            _sysutils.check_dir_write(
                os.path.join(path, "fit_models"), force_create=True
            )
            for i in range(len(self.__fit_models)):
                with open(
                    os.path.join(path, "fit_models", f"split_{i}.pkl"), "wb"
                ) as model_pkl_file:
                    pickle.dump(self.__fit_models[i], model_pkl_file)

        if self.__y_pred != []:
            _sysutils.check_dir_write(os.path.join(path, "y_true"), force_create=True)
            _sysutils.check_dir_write(os.path.join(path, "y_pred"), force_create=True)
            for i in range(len(self.__y_pred)):
                np.savetxt(
                    os.path.join(path, "y_true", f"split_{i}.csv"),
                    self.__y_true[i],
                    delimiter=",",
                )
                np.savetxt(
                    os.path.join(path, "y_pred", f"split_{i}.csv"),
                    self.__y_pred[i],
                    delimiter=",",
                )
        for metric in self:
            metric.export(os.path.join(path, metric.name), *measures, plots=plots)

    def __iter__(self):
        return (
            metric
            for metric in self.__dict__.values()
            if isinstance(metric, BaseMetric)
        )

    def __len__(self):
        return len(
            [
                metric
                for metric in self.__dict__.values()
                if isinstance(metric, BaseMetric)
            ]
        )

    @property
    def model(self):
        return self.__model

    @property
    def fit_models(self):
        return self.__fit_models

    @property
    def y_true(self):
        return self.__y_true

    @property
    def y_pred(self):
        return self.__y_pred
    
    @property
    def feature_names(self):
        return self.__feature_names

    def append_fit_model(self, fit_model):
        self.__fit_models.append(fit_model)

    def __eq__(self, other):
        if not isinstance(other, ModelStats):
            return False
        return (
            self.__model == other.__model
            and self.__fit_models == other.__fit_models
            and all(
                getattr(self, metric) == getattr(other, metric)
                for metric in _constants.METRICS
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
        _sysutils.check_dir_read(path)
        modelcomparison = ModelComparison(models=[], validation=None, X=None, y=None)
        modelcomparison.__splits = []
        for idx in range(len(os.listdir(os.path.join(path, "splits")))):
            general_split_dir = os.path.join(path, "splits", f"split_{idx}")
            train_indices_path = os.path.join(general_split_dir, "train.csv")
            test_indices_path = os.path.join(general_split_dir, "test.csv")

            train = np.loadtxt(train_indices_path, delimiter=",", dtype=int)
            test = np.loadtxt(test_indices_path, delimiter=",", dtype=int)
            modelcomparison.__splits.append((train, test))

        i = 0
        while os.path.isdir(os.path.join(path, f"model_{i}")):
            modelcomparison.__model_stats.append(
                ModelStats.from_path(os.path.join(path, f"model_{i}"))
            )
            i += 1
        return modelcomparison

    def validate(
        self, *pipeline_metrics, keep_models=False, new_splits=False
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

            for train, test in self.__splits:
                fit_model = clone(self.__model_stats[model_idx].model)
                fit_model.fit(self.__X[train], self.__y[train])
                if keep_models:
                    self.__model_stats[model_idx].append_fit_model(fit_model)
                self.__model_stats[model_idx].invoke(
                    fit_model, self.__X, self.__y, train, test
                )

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
            _sysutils.check_dir_write(general_split_dir, force_create=True)
            train_indices_path = os.path.join(general_split_dir, "train.csv")
            test_indices_path = os.path.join(general_split_dir, "test.csv")

            np.savetxt(train_indices_path, train, delimiter=",", fmt="%d")
            np.savetxt(test_indices_path, test, delimiter=",", fmt="%d")

        for model_idx in range(len(self.__model_stats)):
            model = clone(self.__model_stats[model_idx].model)
            model_stats = self.__model_stats[model_idx]
            model_dir = os.path.join(path, f"model_{model_idx}")
            if os.path.exists(model_dir):
                warnings.warn(f"CSV files already exist in {model_dir}. Overwriting.")

            _sysutils.check_dir_write(model_dir, force_create=True)

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
                for metric in model_stats:
                    self.plot(metric_name=metric.name).savefig(
                        os.path.join(path, f"{metric.name} comparison.png")
                    )

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
                    try:
                        from sklearn.metrics import roc_auc_score
                    except ImportError:
                        raise ImportError("sklearn is required for plotting ROC curves")
                    mean_fpr, mean_tpr = metric.mean
                    ax.plot(
                        mean_fpr,
                        mean_tpr,
                        label=f"{str(self.__model_stats[model_idx].model)} Mean ROC curve (AUC = {roc_auc_score(np.concatenate(self.__model_stats[model_idx].y_true), np.concatenate(self.__model_stats[model_idx].y_pred)):.2f})",
                    )
                elif metric_name == "precision_recall_curve":
                    try:
                        from sklearn.metrics import average_precision_score
                    except ImportError:
                        raise ImportError(
                            "sklearn is required for plotting Precision-Recall curves"
                        )
                    mean_precision, mean_recall = metric.mean
                    ax.plot(
                        mean_recall,
                        mean_precision,
                        label=f"{model_idx} Mean Precision-Recall curve (AUC = {
                            average_precision_score(np.concatenate(self.__model_stats[model_idx].y_true), np.concatenate(self.__model_stats[model_idx].y_pred)):.2f}",
                    )
                else:
                    raise ValueError(
                        f"Plotting not supported for metric type {type(metric)}"
                    )
            elif isinstance(metric, ImportanceMetric):
                pass
            else:
                pass
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
