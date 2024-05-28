import numpy as np
import pandas as pd


def assess_intra_group_unique_value_abundance(input_dataframe, target_variable):
    """
    Assess the abundance of unique values of each feature within a target variable, for each unique value of the target variable.
    Then, average the abundance of unique values across all unique values of the target variable.
    Each feature will have a unique value abundance score, which is the average of the abundance of unique values across all unique values of the target variable.
    The higher the score, the more unique values are present within each unique value of the target variable.
    This function is useful for assessing the informativeness of features within a target variable, and can be used to identify features that are more informative than others within a target variable.
    :param input_dataframe: pandas DataFrame with the target variable and features
    :param target_variable: string, the name of the target variable that will be used to group the features
    """
    unique_values = input_dataframe[target_variable].unique()
    unique_values_abundance_list = []

    for value in unique_values:
        current_subset = input_dataframe[
            input_dataframe[target_variable] == value
        ].drop(columns=[target_variable])
        current_subset_unique_values_percentage = (
            current_subset.nunique() / current_subset.shape[0]
        )
        unique_values_abundance_list.append(current_subset_unique_values_percentage)

    unique_values_abundance_df = pd.concat(
        unique_values_abundance_list, axis=1
    ).transpose()
    average_unique_values = np.mean(unique_values_abundance_df, axis=0)
    return average_unique_values
