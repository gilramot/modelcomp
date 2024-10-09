import pickle
import numpy as np
from tqdm import tqdm
import modelcomp.utils as utils
import warnings
from modelcomp.constants import METRICS
import os
import pandas as pd
import pickle as pkl

__all__ = ["ModelComparison"]




class Metric:
    def __init__(self, name):
        self.__name = name

        if name not in METRICS:
            warnings.warn(f"Metric {name} not supported")
            return None

        if name in ["precision_recall_curve", "roc_curve"]:
            self.type = "curve"
            if name == "precision_recall_curve":
                self.y_true = []
                self.y_pred = []
        elif name in ["accuracy_score", "precision_score", "recall_score", "f1_score"]:
            self.type = "score"
        else:
            self.type = "other"

        self.values = []

    def invoke(self, model, X, y, train, test):

        if self.type == "curve":
            y_pred = model.predict_proba(X[test])[:, 1]
            if self.name == "precision_recall_curve":
                precision, recall, thresholds = METRICS[self.name](y[test], y_pred)
                self.y_true.append(np.array(y[test]))
                self.y_pred.append(np.array(y_pred))
                self.append(np.vstack([precision, recall, np.append(thresholds, 0)]))
            elif self.name == "roc_curve":
                fpr, tpr, thresholds = METRICS[self.name](y[test], y_pred)
                self.append(np.vstack([fpr, tpr, thresholds]))
            else:
                pass


        elif self.type == "score":
            self.append(METRICS[self.name](y[test], model.predict(X[test])))

        else:
            pass

    def append(self, value):
        if self.type == "score":
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                value = value.item()
            if isinstance(value, (list, np.ndarray, pd.DataFrame)):
                raise ValueError(
                    f"Expected 1 value for metric {self.name}, got {len(value)}"
                )
            self.values.append(value)
        elif self.type == "curve":
            if len(value) != 3:
                raise ValueError(
                    f"Expected 3 values for metric {self.name}, got {len(value)}"
                )
            self.values.append(value)
        else:
            pass

    def export(self, path, *measures):
        os.makedirs(os.path.join(path, f"{self.name}"), exist_ok=True)
        if self.type == "score":
            for index, value in enumerate(self.values):
               np.savetxt(os.path.join(path, f"{self.name}", f"split_{index}.csv"), [value], delimiter=",")
        elif self.type == "curve":
            for index, values in enumerate(self.values):
                np.savetxt(os.path.join(path, f"{self.name}", f"split_{index}.csv"), values, delimiter=",")
                if self.name == "precision_recall_curve":
                    os.makedirs(os.path.join(path, "y_true"), exist_ok=True)
                    np.savetxt(os.path.join(path, "y_true", f"split_{index}.csv"), self.y_true[index].astype(int), delimiter=",")
                    os.makedirs(os.path.join(path, "y_pred"), exist_ok=True)
                    np.savetxt(os.path.join(path, "y_pred", f"split_{index}.csv"), self.y_pred[index].astype(int), delimiter=",")
        else:
            pass
            
        for measure in measures:
            if measure == "mean":
                if isinstance(self.mean, np.ndarray) and self.mean.shape != ():
                    np.savetxt(os.path.join(path, self.name, "mean.csv"), self.mean, delimiter=",")
                elif isinstance(self.mean, tuple):
                    np.savetxt(os.path.join(path, self.name, "mean.csv"), np.vstack(self.mean))
                elif self.mean is not None:
                    np.savetxt(os.path.join(path, self.name, "mean.csv"), [self.mean], delimiter=",")
            elif measure == "std":
                if isinstance(self.std, np.ndarray) and self.std.shape != ():
                    np.savetxt(os.path.join(path, self.name, "srd.csv"), self.std, delimiter=",")
                elif isinstance(self.std, tuple):
                    np.savetxt(os.path.join(path, self.name, "std.csv"), np.vstack(self.std))
                elif self.std is not None:
                    np.savetxt(os.path.join(path, self.name, "std.csv"), [self.std], delimiter=",")
            else:
                pass


    @property
    def mean(self):
        if self.type == "curve":
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
        elif self.type == "score":
            return np.mean(self.values)
        else:
            pass
    
    @property
    def std(self):
        if self.type == "curve":
            if self.name == "roc_curve":
                interp_values = []
                for values in self.values:
                    mean_tpr = np.interp(np.linspace(0, 1, 101), values[0], values[1])
                    mean_tpr[0] = 0.0
                    mean_tpr[-1] = 1.0
                    interp_values.append(mean_tpr)
                return np.linspace(0, 1, 101), np.std(interp_values, axis=0)
            elif self.name == "precision_recall_curve":
                pass
        elif self.type == "score":
            return np.std(self.values)
        else:
            pass

    @property
    def name(self):
        return self.__name

    def __getitem__(self, index):
        return self.values[index]


class ModelStats:
    def __init__(self, model, *metrics):
        self.__model = model
        self.__fit_models = []
        for metric in metrics:
            metric_to_set = Metric(name=metric)
            if metric_to_set is not None:
                setattr(self, metric, metric_to_set)

    def append_metrics(self, **metrics):
        for metric in metrics.keys():
            if metric not in METRICS:
                raise ValueError(f"{metric} not found in METRICS")
            if not hasattr(self, metric):
                setattr(self, metric, Metric(name=metric))

            getattr(self, metric).append(metrics[metric])
    
    def export(self, path, *measures):
        for metric in self:
            metric.export(path, *measures)

    def __iter__(self):
        return (
            metric for metric in self.__dict__.values() if isinstance(metric, Metric)
        )

    def __len__(self):
        return len(
            [metric for metric in self.__dict__.values() if isinstance(metric, Metric)]
        )

    @property
    def model(self):
        return self.__model

    @property
    def fit_models(self):
        return self.__fit_models

    def append_fit_model(self, fit_model):
        self.__fit_models.append(fit_model)
    


class ModelComparison:
    def __init__(self, models=[], validation=None, X=None, y=None, feature_names=None):
        self.model_stats = {model: None for model in models}
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

    def validate(self, *pipeline_metrics, keep_models=False, models=None, new_splits=False):
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
                warnings.warn("Train and test sets do not exist and new_splits=False. Creating new splits anyway.")
                self.__splits = []
                for train, test in self.__validation.split(self.__X, self.__y):
                    self.__splits.append((train, test))

        for model in self.model_stats.keys():
            
            if self.model_stats[model] is not None:
                warnings.warn(
                    f"Metrics for model {model} already exist and will be overwritten."
                )

            self.model_stats[model] = ModelStats(model, *pipeline_metrics)

            for train, test in tqdm(self.__splits, desc="Split"):
                fit_model = model.fit(self.__X[train], self.__y[train])
                if keep_models:
                    self.model_stats[model].append_fit_model(fit_model)
                # fitting model

                for metric in self.model_stats[model]:
                    metric.invoke(fit_model, self.__X, self.__y, train, test)

    def explain(self):
        for model in tqdm(self.model_stats, desc="Model"):
            model_stats = self.model_stats[model]
            if hasattr(model_stats, "builtin_importance"):
                warnings.warn(
                    f"Feature importance values for model {model} already exist and will be overwritten."
                )

            model_stats.builtin_importance = [None] * len(model_stats.fit_models)
            model_stats.shap_values = [None] * len(model_stats.fit_models)
            model_stats.shap_explainers = [None] * len(model_stats.fit_models)
            for split_index, (train, test) in enumerate(self.__splits):
                (
                    model_stats.builtin_importance[split_index],
                    model_stats.shap_explainers[split_index],
                    model_stats.shap_values[split_index],
                ) = utils.get_explainers(
                    model_stats.fit_models[split_index],
                    self.__X[test],
                    self.__feature_names,
                )
            feature_importances = model_stats.builtin_importance[0]
            for feature_importances_in_fold in model_stats.builtin_importance[1:]:
                feature_importances = feature_importances.add(
                    abs(feature_importances_in_fold), fill_value=0
                )
            feature_importances["Importance"] = feature_importances["Importance"].map(
                lambda old_value: old_value / len(model_stats.builtin_importance)
            )

            model_stats.mean_builtin_importance = feature_importances

            shap_values = model_stats.shap_values[0].values
            for shap_values_in_fold in model_stats.shap_values[1:]:
                shap_values = shap_values.add(
                    abs(shap_values_in_fold.values), fill_value=0
                )
            shap_values = shap_values.map(
                lambda old_value: old_value / len(model_stats.shap_values)
            )
            model_stats.mean_shap_values = shap_values

    def remove_model(self, model, keep_data=False):
        self.models.remove(model)
        if not keep_data:
            del self.model_stats[model]


    
    def export(self, path, *measures):
        for idx, (train, test) in enumerate(self.splits):
            general_split_dir = os.path.join(path, "splits", f"split_{idx}")
            os.makedirs(general_split_dir, exist_ok=True)
            train_indices_path = os.path.join(general_split_dir, "train.csv")
            test_indices_path = os.path.join(general_split_dir, "test.csv")

            np.savetxt(train_indices_path, train, delimiter=",", fmt="%d")
            np.savetxt(test_indices_path, test, delimiter=",", fmt="%d")

        for model_index, model in enumerate(self.model_stats):
            model_stats = self.model_stats[model]
            model_dir = os.path.join(path, f"model_{model_index}")
            os.makedirs(model_dir, exist_ok=True)

            np.savetxt(
                os.path.join(model_dir, "name.txt"), [str(model)], delimiter=",", fmt="%s"
            )

            pkl.dump(model, open(os.path.join(model_dir, "model.pkl"), "wb"))

            if os.path.exists(model_dir):
                warnings.warn(
                    f"CSV files for model {model} already exist in {model_dir}. Overwriting."
                )
            for split_index, (train, test) in enumerate(self.splits):

                if model_stats.fit_models:
                    model_pkl_path = os.path.join(model_dir, "fit_models", f"split_{split_index}.pkl")
                    with open(model_pkl_path, "wb") as model_pkl_file:
                        pickle.dump(model_stats.fit_models[split_index], model_pkl_file)
                
                model_stats.export(model_dir, *measures)
                    
    def load(self, path):
        for idx in range(len(os.listdir(os.path.join(path, "splits")))):
            general_split_dir = os.path.join(path, "splits", f"split_{idx}")
            train_indices_path = os.path.join(general_split_dir, "train.csv")
            test_indices_path = os.path.join(general_split_dir, "test.csv")

            train = np.loadtxt(train_indices_path, delimiter=",", dtype=int)
            test = np.loadtxt(test_indices_path, delimiter=",", dtype=int)
            self.__splits.append((train, test))

        for model_index in range(len(os.listdir(path)) - 1):
            model_dir = os.path.join(path, f"model_{model_index}")
            model_name_path = os.path.join(model_dir, "name.txt")
            model_pkl_path = os.path.join(model_dir, "model.pkl")

            with open(model_name_path, "r") as model_name_file:
                model_name = model_name_file.read().strip()

            with open(model_pkl_path, "rb") as model_pkl_file:
                model = pickle.load(model_pkl_file)

            self.model_stats[model] = ModelStats(model)

            fit_models_dir = os.path.join(model_dir, "fit_models")
            if os.path.exists(fit_models_dir):
                for split_index in range(len(self.__splits)):
                    fit_model_pkl_path = os.path.join(fit_models_dir, f"split_{split_index}.pkl")
                    with open(fit_model_pkl_path, "rb") as fit_model_pkl_file:
                        fit_model = pickle.load(fit_model_pkl_file)
                    self.model_stats[model].append_fit_model(fit_model)

            for metric in METRICS.keys():
                metric_dir = os.path.join(model_dir, metric)
                if os.path.exists(metric_dir):
                    metric_instance = Metric(name=metric)
                    for split_index in range(len(self.__splits)):
                        metric_csv_path = os.path.join(metric_dir, f"split_{split_index}.csv")
                        metric_values = pd.read_csv(metric_csv_path).values
                        metric_instance.append(metric_values)
                        setattr(self.model_stats[model], metric, metric_instance)
        
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
        return self.model_stats.keys()

    @models.setter
    def models(self, models):
        for model in models:
            if model not in self.model_stats:
                self.model_stats[model] = None
        for model in self.model_stats:
            if model not in models:
                del self.model_stats[model]

    def add_model(self, model):
        self.model_stats[model] = None

    def remove_model(self, model):
        del self.model_stats[model]

    
