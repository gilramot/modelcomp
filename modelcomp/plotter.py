import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import modelcomp as mc


def individual_plots(save_to_unjoined):
    """
    Plot the performance of a single model
    :param save_to_unjoined: Location to save the plots to
    :return: Plots showing the performance of a single model, exported to the filesystem (default: 'export/<train label>/<test label>/<model name>/plots')
    """
    save_to = mc.join_save(os.path.join(save_to_unjoined, "plots"))
    trained_on, tested_on, model_name = mc.filename_to_data(save_to)
    trained_on = "all labels" if trained_on == "a" else trained_on
    tested_on = "all labels" if tested_on == "a" else tested_on
    title_start = f"{model_name} " + (
        f"(trained on {trained_on}, tested on {tested_on}) - "
        if not trained_on == "multiclass"
        else "(multiclass) - "
    )
    (
        _,
        interp_tpr,
        fprs,
        tprs,
        interp_recall,
        precisions,
        recalls,
        aucs,
        pr_aucs,
        feature_importances,
        shap_values,
    ) = mc.read_data(save_to_unjoined)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"AUC = %0.2f " % (round(mean_auc, 2))
        + ("±%0.2f" % (round(std_auc, 2)) if trained_on == tested_on else ""),
        alpha=0.8,
    )
    # roc curve

    std_tpr = np.std(interp_tpr, axis=0)
    if max(std_tpr) > 0:
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.3,
            label="±1 std. dev.",
        )
    # roc standard deviation

    plt.xlim = [-0.05, 1.05]
    plt.ylim = ([-0.05, 1.05],)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_start} ROC curve")
    if interp_tpr is not None:
        for rate_index in range(len(interp_tpr)):
            plt.plot(mean_fpr, interp_tpr[rate_index], alpha=0.15)
            # individual splits' roc

    plt.plot([0, 1], [0, 1], linestyle="--")
    # randomness line

    plt.legend(loc="lower right")
    # legend location

    mc.write_plot(save_to, "roc")
    # saving plot

    mean_recall = np.mean(interp_recall, axis=0)
    mean_auc = np.mean(pr_aucs)
    std_auc = np.std(pr_aucs)
    plt.plot(
        mean_fpr,
        mean_recall,
        color="b",
        label=r"PR AUC = %0.2f " % (round(mean_auc, 2))
        + ("±%0.2f" % (round(std_auc, 2)) if trained_on == tested_on else ""),
        alpha=0.8,
    )
    # pr curve
    std_recall = np.std(interp_recall, axis=0)
    if max(std_recall) > 0:
        recalls_upper = np.minimum(mean_recall + std_recall, 1)
        recalls_lower = np.maximum(mean_recall - std_recall, 0)
        plt.fill_between(
            mean_fpr,
            recalls_lower,
            recalls_upper,
            color="grey",
            alpha=0.3,
            label="±1 std. dev.",
        )
    # pr standard deviation

    plt.xlim = ([-0.05, 1.05],)
    plt.ylim = [-0.05, 1.05]
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_start} PR Curve")

    if interp_recall is not None:
        for rate_index in range(len(interp_recall)):
            plt.plot(mean_fpr, interp_recall[rate_index], alpha=0.15)

    plt.legend(loc="lower left")
    # legend location

    mc.write_plot(save_to, "precision-recall")
    # saving plot

    if feature_importances is not None:
        feature_importances.sort_values(by="Importance", ascending=False, inplace=True)
        important_features = feature_importances[:20]
        plt.barh(important_features.index.values, important_features["Importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"{title_start} Feature Importance")
        mc.write_plot(save_to, "feature_importance")
    # plotting feature importance
    if shap_values is not None:
        important_features = shap_values[:20]
        plt.barh(important_features.index.values, important_features["avg_shap_value"])
        plt.gca().invert_yaxis()
        plt.xlabel("Impact on model output")
        plt.ylabel("Feature")
        plt.title(f"{title_start} SHAP Values")
        mc.write_plot(save_to, "shap_values")
    # plotting shap values


def general_plots(positive_uniques):
    """
    Plots comparing the models' performances
    :param positive_uniques: The positive diseases' IDs in the input data
    :return: Plots exported to the filesystem (default: 'export/GENERAL PLOTS')
    """
    model_names = mc.join_save(os.path.join("A", "A")).listdir()
    save_to = mc.join_save("GENERAL PLOTS")
    mc.make_dir(save_to)
    model_aucs = []
    for (model_name,) in model_names:
        mean_auc = np.mean(
            np.genfromtxt(
                mc.join_save(os.path.join("A", "A", model_name, "data", "aucs.csv"))
            )
        )
        model_aucs.append([model_name, mean_auc, "A", "A", True])
    for tested_on in positive_uniques:
        for trained_on in positive_uniques:
            for model_name in model_names:
                mean_auc = np.mean(
                    np.genfromtxt(
                        mc.join_save(
                            os.path.join(
                                trained_on, tested_on, model_name, "data", "aucs.csv"
                            )
                        )
                    )
                )
                model_aucs.append(
                    [
                        model_name,
                        mean_auc,
                        trained_on,
                        tested_on,
                        trained_on == tested_on,
                    ]
                )
    model_aucs_df = pd.DataFrame(
        model_aucs,
        columns=["Model", "AUC", "Train Data", "Test Data", "Train/Test equality"],
    )
    average_model_aucs = list()
    for model_name in model_names:
        average_model_aucs.append(
            np.mean(model_aucs_df["AUC"].loc[model_aucs_df["Model"] == model_name])
        )
    order_by = [model_names[i] for i in np.argsort(average_model_aucs)[::-1]]
    sns.catplot(
        data=model_aucs_df,
        x="Model",
        y="AUC",
        kind="swarm",
        order=order_by,
        hue="Train/Test equality",
    )
    sns.boxplot(
        data=model_aucs_df,
        x="Model",
        y="AUC",
        order=order_by,
        showcaps=False,
        boxprops={"facecolor": "None"},
        whiskerprops={"linewidth": 0},
    )
    plt.xticks(fontsize=8)
    plt.title("Model comparison (AUC)")
    mc.write_plot(save_to, "model_comparison_auc")
    np.savetxt(
        os.path.join(save_to, "average_model_aucs.csv"),
        np.array([model_names, average_model_aucs]),
        delimiter=",",
        fmt="%s",
    )

    max_auc_annot = model_aucs_df[["Train Data", "Test Data", "AUC"]].copy().fillna(0)
    max_auc_annot = max_auc_annot.loc[max_auc_annot["Train Data"] != "A"].loc[
        max_auc_annot["Test Data"] != "A"
    ]
    max_auc_annot = max_auc_annot.loc[
        max_auc_annot.groupby(["Train Data", "Test Data"])["AUC"].idxmax()
    ]
    max_auc_annot = max_auc_annot.pivot(
        index="Train Data", columns="Test Data", values="AUC"
    )

    max_auc = model_aucs_df[["Train Data", "Test Data", "AUC"]].copy().dropna()
    max_auc = max_auc.loc[max_auc["Train Data"] != "A"].loc[max_auc["Test Data"] != "A"]
    max_auc = max_auc.loc[max_auc.groupby(["Train Data", "Test Data"])["AUC"].idxmax()]
    max_auc = max_auc.pivot(index="Train Data", columns="Test Data", values="AUC")

    sns.heatmap(max_auc, cmap="viridis", vmin=0.4, vmax=1)
    for i, col in enumerate(max_auc.columns):
        for j, row in enumerate(max_auc.columns):
            plt.annotate(
                str(
                    round(max_auc_annot[row][col], 2)
                    if max_auc_annot[row][col] != 0
                    else "NaN"
                ),
                xy=(j + 0.5, i + 0.5),
                ha="center",
                va="center",
                color="white" if max_auc_annot[row][col] != 0 else "black",
            )
    plt.title("AUCs of most accurate model for each train/test data")
    mc.write_plot(save_to, "correlation_heatmap_auc")

    max_auc = model_aucs_df[["Train Data", "Test Data", "Model", "AUC"]].copy().dropna()
    max_auc = max_auc.loc[max_auc["Train Data"] != "A"].loc[max_auc["Test Data"] != "A"]
    max_auc = max_auc.loc[max_auc.groupby(["Train Data", "Test Data"])["AUC"].idxmax()]
    max_auc = max_auc.drop("AUC", axis=1).pivot(
        index="Train Data", columns="Test Data", values="Model"
    )
    sns.heatmap(
        max_auc.replace(order_by[::-1], list(range(1, len(model_names) + 1))),
        cbar=False,
        cmap="tab10",
    )
    for i, col in enumerate(max_auc.columns):
        for j, row in enumerate(max_auc.columns):
            plt.annotate(
                (max_auc[row][col] if not str(max_auc[row][col]) == "nan" else "N/A"),
                xy=(j + 0.5, i + 0.5),
                ha="center",
                va="center",
                color="white" if not str(max_auc[row][col]) == "nan" else "black",
            )
    plt.title("Most accurate model for each train/test data (AUC)")
    mc.write_plot(save_to, "correlation_heatmap_models_auc")

    model_pr_aucs = []
    for model_name, model_name_short in model_names:
        mean_pr_auc = np.mean(
            np.genfromtxt(
                mc.join_save(os.path.join("A", "A", model_name, "data", "pr_aucs.csv"))
            )
        )
        model_pr_aucs.append([model_name_short, mean_pr_auc, "A", "A", True])
    for tested_on in positive_uniques:
        for trained_on in positive_uniques:
            for model_name, model_name_short in model_names:
                mean_pr_auc = np.mean(
                    np.genfromtxt(
                        mc.join_save(
                            os.path.join(
                                trained_on, tested_on, model_name, "data", "pr_aucs.csv"
                            )
                        )
                    )
                )
                model_pr_aucs.append(
                    [
                        model_name_short,
                        mean_pr_auc,
                        trained_on,
                        tested_on,
                        trained_on == tested_on,
                    ]
                )
    model_pr_aucs_df = pd.DataFrame(
        model_pr_aucs,
        columns=["Model", "PR AUC", "Train Data", "Test Data", "Train/Test equality"],
    )
    average_model_pr_aucs = list()
    for model_name in model_names:
        average_model_pr_aucs.append(
            np.mean(
                model_pr_aucs_df["PR AUC"].loc[model_pr_aucs_df["Model"] == model_name]
            )
        )
    pr_order_by = [model_names[i] for i in np.argsort(average_model_pr_aucs)[::-1]]
    sns.catplot(
        data=model_pr_aucs_df,
        x="Model",
        y="PR AUC",
        kind="swarm",
        order=pr_order_by,
        hue="Train/Test equality",
    )
    sns.boxplot(
        data=model_pr_aucs_df,
        x="Model",
        y="PR AUC",
        order=pr_order_by,
        showcaps=False,
        boxprops={"facecolor": "None"},
        whiskerprops={"linewidth": 0},
    )
    plt.xticks(fontsize=8)
    plt.title("Model comparison (PR AUC)")
    mc.write_plot(save_to, "model_comparison_pr_auc")
    np.savetxt(
        os.path.join(save_to, "average_model_pr_aucs.csv"),
        np.array([model_names, average_model_pr_aucs]),
        delimiter=",",
        fmt="%s",
    )

    max_pr_auc_annot = (
        model_pr_aucs_df[["Train Data", "Test Data", "PR AUC"]].copy().fillna(0)
    )
    max_pr_auc_annot = max_pr_auc_annot.loc[max_pr_auc_annot["Train Data"] != "A"].loc[
        max_pr_auc_annot["Test Data"] != "A"
    ]
    max_pr_auc_annot = max_pr_auc_annot.loc[
        max_pr_auc_annot.groupby(["Train Data", "Test Data"])["PR AUC"].idxmax()
    ]
    max_pr_auc_annot = max_pr_auc_annot.pivot(
        index="Train Data", columns="Test Data", values="PR AUC"
    )

    max_pr_auc = model_pr_aucs_df[["Train Data", "Test Data", "PR AUC"]].copy().dropna()
    max_pr_auc = max_pr_auc.loc[max_pr_auc["Train Data"] != "A"].loc[
        max_pr_auc["Test Data"] != "A"
    ]
    max_pr_auc = max_pr_auc.loc[
        max_pr_auc.groupby(["Train Data", "Test Data"])["PR AUC"].idxmax()
    ]
    max_pr_auc = max_pr_auc.pivot(
        index="Train Data", columns="Test Data", values="PR AUC"
    )

    sns.heatmap(max_pr_auc, cmap="viridis", vmin=0.4, vmax=1)
    for i, col in enumerate(max_pr_auc.columns):
        for j, row in enumerate(max_pr_auc.columns):
            plt.annotate(
                str(
                    round(max_pr_auc_annot[row][col], 2)
                    if max_pr_auc_annot[row][col] != 0
                    else "NaN"
                ),
                xy=(j + 0.5, i + 0.5),
                ha="center",
                va="center",
                color="white" if max_pr_auc_annot[row][col] != 0 else "black",
            )
    plt.title("PR AUCs of most accurate model for each train/test data")
    mc.write_plot(save_to, "correlation_heatmap_pr_auc")

    max_pr_auc = (
        model_pr_aucs_df[["Train Data", "Test Data", "Model", "PR AUC"]].copy().dropna()
    )
    max_pr_auc = max_pr_auc.loc[max_pr_auc["Train Data"] != "A"].loc[
        max_pr_auc["Test Data"] != "A"
    ]
    max_pr_auc = max_pr_auc.loc[
        max_pr_auc.groupby(["Train Data", "Test Data"])["PR AUC"].idxmax()
    ]
    max_pr_auc = max_pr_auc.drop("PR AUC", axis=1).pivot(
        index="Train Data", columns="Test Data", values="Model"
    )
    sns.heatmap(
        max_pr_auc.replace(order_by[::-1], list(range(1, len(model_names) + 1))),
        cbar=False,
        cmap="tab10",
    )
    for i, col in enumerate(max_pr_auc.columns):
        for j, row in enumerate(max_pr_auc.columns):
            plt.annotate(
                (
                    max_pr_auc[row][col]
                    if not str(max_pr_auc[row][col]) == "nan"
                    else "N/A"
                ),
                xy=(j + 0.5, i + 0.5),
                ha="center",
                va="center",
                color="white" if not str(max_pr_auc[row][col]) == "nan" else "black",
            )
    plt.title("Most accurate model for each train/test data (PR AUC)")
    mc.write_plot(save_to, "correlation_heatmap_models_pr_auc")

    model_accuracies = []
    for model_name, model_name_short in model_names:
        mean_auc = np.mean(
            np.genfromtxt(
                mc.join_save(
                    os.path.join("A", "A", model_name, "data", "accuracies.csv")
                )
            )
        )
        model_accuracies.append([model_name_short, mean_auc, "A", "A", True])
    for tested_on in positive_uniques:
        for trained_on in positive_uniques:
            for model_name, model_name_short in model_names:
                mean_auc = np.mean(
                    np.genfromtxt(
                        mc.join_save(
                            os.path.join(
                                trained_on,
                                tested_on,
                                model_name,
                                "data",
                                "accuracies.csv",
                            )
                        )
                    )
                )
                model_accuracies.append(
                    [
                        model_name_short,
                        mean_auc,
                        trained_on,
                        tested_on,
                        trained_on == tested_on,
                    ]
                )
    model_accuracies_df = pd.DataFrame(
        model_accuracies,
        columns=["Model", "Accuracy", "Train Data", "Test Data", "Train/Test equality"],
    )
    average_model_accuracies = list()
    for model_name in model_names:
        average_model_accuracies.append(
            np.mean(
                model_accuracies_df["Accuracy"].loc[
                    model_accuracies_df["Model"] == model_name
                ]
            )
        )
    order_by = [model_names[i] for i in np.argsort(average_model_accuracies)[::-1]]
    sns.catplot(
        data=model_accuracies_df,
        x="Model",
        y="Accuracy",
        kind="swarm",
        order=order_by,
        hue="Train/Test equality",
    )
    sns.boxplot(
        data=model_accuracies_df,
        x="Model",
        y="Accuracy",
        order=order_by,
        showcaps=False,
        boxprops={"facecolor": "None"},
        whiskerprops={"linewidth": 0},
    )
    plt.xticks(fontsize=8)
    plt.title("Model comparison (Accuracy)")
    mc.write_plot(save_to, "model_comparison_accuracy")
    np.savetxt(
        os.path.join(save_to, "average_model_accuracies.csv"),
        np.array([model_names, average_model_accuracies]),
        delimiter=",",
        fmt="%s",
    )
