import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sparc_multiomics.constants import RANDOM_SEED

random.seed(RANDOM_SEED)


def model_evaluation(
    input_class,
    input_predictions,
    verbose=False,
    write_mode=False,
):
    """
    Perform model evaluation using a vector of the actual class and the predicted class
    ACTUAL CLASS MUST BE FIRST ARGUMENT
    :param input_class: The actual class
    :param input_predictions: The predicted class
    :param verbose: If True, print the performance metrics
    :param write_mode: If not False, write the performance metrics to a file
    """
    if write_mode is not False:
        output_file_name = f"results/{write_mode}_performance.csv"
    unique_values = list(np.unique(input_class))
    if len(unique_values) == 2:
        accuracy = accuracy_score(input_class, input_predictions)
        precision = precision_score(input_class, input_predictions)
        recall = recall_score(input_class, input_predictions)
        auc = roc_auc_score(input_class, input_predictions)
        f1_value = f1_score(input_class, input_predictions)
    elif len(unique_values) > 2:
        accuracy = accuracy_score(input_class, input_predictions)
        precision = np.mean(
            precision_score(input_class, input_predictions, average="weighted")
        )
        recall = np.mean(
            recall_score(input_class, input_predictions, average="weighted")
        )
        auc = 0.0  # DEBUG
        f1_value = np.mean(f1_score(input_class, input_predictions, average="weighted"))

    if verbose:
        print(
            f"Accuracy:{round(accuracy, 2)}\n",
            f"Precision:{round(precision, 2)}\n",
            f"Recall:{round(recall, 2)}\n",
            f"AUC:{round(auc, 2)}\n",
            f"F1-score:{round(f1_value, 2)}\n",
        )
    if write_mode is not False:
        confusion_matrix_values = pd.DataFrame(
            confusion_matrix(input_class, input_predictions),
            columns=unique_values,
            index=unique_values,
        )
        confusion_matrix_values.to_csv(f"results/{write_mode}_confusion_matrix.csv")

        with open(output_file_name, "w") as output_file:
            output_file.write("Metric,Value\n")
            output_file.write("Accuracy," + str(accuracy) + "\n")
            output_file.write("Precision," + str(precision) + "\n")
            output_file.write("Recall," + str(recall) + "\n")
            output_file.write("AUC," + str(auc) + "\n")
            output_file.write("F1-score," + str(f1_value) + "\n")
    return [accuracy, precision, recall, auc, f1_value]


def fold_evaluation(input_performance, write_mode=False):
    """
    Compute the mean and standard deviation of the performance metrics, in the context of a nested cross-validation
    :param input_performance: The performance metrics from the nested cross-validation
    :param write_mode: The write mode, if not False, write the performance metrics to a file
    """
    mean_accuracy, accuracy_deviation = (
        np.mean(input_performance["accuracy"]),
        np.std(input_performance["accuracy"]),
    )
    mean_precision, precision_deviation = (
        np.mean(input_performance["precision"]),
        np.std(input_performance["precision"]),
    )
    mean_recall, recall_deviation = (
        np.mean(input_performance["recall"]),
        np.std(input_performance["recall"]),
    )
    mean_auc, auc_deviation = (
        np.mean(input_performance["auc"]),
        np.std(input_performance["auc"]),
    )
    mean_f1, f1_deviation = (
        np.mean(input_performance["f1"]),
        np.std(input_performance["f1"]),
    )

    if write_mode is not False:
        print(f"==={write_mode} performance")
    else:
        print("===Performance===")
    print(f"Mean Accuracy: {mean_accuracy:.2f} (± {accuracy_deviation:.2f})")
    print(f"Mean Precision: {mean_precision:.2f} (± {precision_deviation:.2f})")
    print(f"Mean Recall: {mean_recall:.2f} (± {recall_deviation:.2f})")
    print(f"Mean AUC: {mean_auc:.2f} (± {auc_deviation:.2f})")
    print(f"Mean F1-score: {mean_f1:.2f} (± {f1_deviation:.2f})")
    if write_mode is not False:
        with open(f"results/{write_mode}_nested_performance.csv", "w") as output_file:
            output_file.write("Metric,Mean,Standard Deviation\n")
            output_file.write(f"Accuracy,{mean_accuracy},{accuracy_deviation}\n")
            output_file.write(f"Precision,{mean_precision},{precision_deviation}\n")
            output_file.write(f"Recall,{mean_recall},{recall_deviation}\n")
            output_file.write(f"AUC,{mean_auc},{auc_deviation}\n")
            output_file.write(f"F1-score,{mean_f1},{f1_deviation}\n")


def group_wise_split(input_data, group_column_name, split_ratio=0.8):
    """
    Split the data into training and testing sets based on the groups
    :param input_data: The input data
    :param group_column_name: The column name containing the groups
    :param split_ratio: The split ratio, default is 0.8
    """
    unique_groups = input_data[group_column_name].unique()
    random.shuffle(unique_groups)
    train_groups = unique_groups[: int(split_ratio * len(unique_groups))]
    test_groups = unique_groups[int(split_ratio * len(unique_groups)) :]

    train_indexes = input_data.loc[
        input_data[group_column_name].isin(train_groups)
    ].index
    test_indexes = input_data.loc[input_data[group_column_name].isin(test_groups)].index

    return train_indexes, test_indexes


def train_model(
    current_method,
    input_data,
    writing_tag="",
    identifier_column_tag=None,
):
    """
    Function to train a model using nested cross-validation
    :param current_method: The current method to use
    :param input_data: The input data, a dictionary containing the features, labels, and ids, in the format. ids is optional
        {"features": features_dataframe, "labels": labels_dataframe, "ids": ids_dataframe}
    :param writing_tag: The writing tag to use
    :param identifier_column_tag: The identifier column tag to use, if None, use the index

    :return: The best estimator, the predicted values, the feature importances, and the scaler
    """

    random.seed(RANDOM_SEED)
    if identifier_column_tag is not None:
        train_groups_indexes, test_groups_indexes = group_wise_split(
            input_data["ids"], group_column_name=identifier_column_tag, split_ratio=0.8
        )

    else:
        train_groups_indexes = random.sample(
            range(input_data["features"].shape[0]),
            int(0.8 * input_data["features"].shape[0]),
        )
        test_groups_indexes = list(
            set(range(input_data["features"].shape[0])) - set(train_groups_indexes)
        )

    features_outer_train = input_data["features"].loc[train_groups_indexes]
    features_outer_test = input_data["features"].loc[test_groups_indexes]
    labels_outer_train = input_data["labels"].loc[train_groups_indexes]
    labels_outer_test = input_data["labels"].loc[test_groups_indexes]
    # labels_outer_test.to_csv("results/labels_outer_test.csv")

    # Normalize the features on the train data
    scaler = StandardScaler()
    scaler.fit(features_outer_train)
    normalized_features_outer_train = scaler.transform(features_outer_train)
    normalized_features_outer_test = scaler.transform(features_outer_test)

    # Define inner and outer folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Perform nested cross-validation
    outer_scores, test_scores = [], []
    feature_importances = []

    # Iter over the outer folder
    for train_idx, val_idx in outer_cv.split(
        normalized_features_outer_train, labels_outer_train
    ):
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        # Grid Search on inner folds
        grid_search = GridSearchCV(
            estimator=current_method[0],
            param_grid=current_method[1],
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        # for every inner CV, get the train (where we do grid search) and validation splits
        features_train_fold, features_val_fold = (
            normalized_features_outer_train[train_idx],
            normalized_features_outer_train[val_idx],
        )
        labels_train_fold, labels_val_fold = (
            labels_outer_train.values[train_idx],
            labels_outer_train.values[val_idx],
        )

        # Fit the model with hyperparameter tuning on the training fold
        grid_search.fit(features_train_fold, labels_train_fold)

        # Get the best estimator from the inner loop
        best_estimator = grid_search.best_estimator_

        # Evaluate the model on the validation fold
        predicted_values = best_estimator.predict(features_val_fold)
        current_performance = model_evaluation(
            labels_val_fold,
            predicted_values,
            verbose=False,
        )
        predicted_test_values = best_estimator.predict(normalized_features_outer_test)
        current_test_performance = model_evaluation(
            labels_outer_test,
            predicted_test_values,
            verbose=False,
        )
        feature_importances.append(best_estimator.feature_importances_)

        # Get the best parameters from the inner loop
        best_parameters = grid_search.best_params_
        print(f"Best parameters: {best_parameters}")

        outer_scores.append(current_performance)
        test_scores.append(current_test_performance)

    performance_dataframe = pd.DataFrame(
        outer_scores, columns=["accuracy", "precision", "recall", "auc", "f1"]
    )
    performance_test_dataframe = pd.DataFrame(
        test_scores, columns=["accuracy", "precision", "recall", "auc", "f1"]
    )

    # Compute the mean score and standard deviation from the outer loop
    if writing_tag is not False:
        fold_evaluation(performance_dataframe, write_mode=f"{writing_tag}_validation")
        fold_evaluation(
            performance_test_dataframe,
            write_mode=f"{writing_tag}_testing",
        )
    else:
        fold_evaluation(performance_dataframe, write_mode="performance_validation")
        fold_evaluation(performance_test_dataframe, write_mode="performance_testing")

    # Calculate the average feature importance
    feature_importances = np.mean(np.array(feature_importances), axis=0)

    # Finally, evaluate the best model on the test set
    predicted_test = best_estimator.predict(normalized_features_outer_test)

    return best_estimator, predicted_test, feature_importances, scaler
