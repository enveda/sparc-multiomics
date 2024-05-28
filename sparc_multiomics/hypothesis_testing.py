import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


class HypothesisTesting:
    """
    Pass in the input data, the variable type, the target column, the group column, and the target group.
    The variable type can be either "continuous" or "categorical". The p-value will be calculated based on the variable type.
    :param input_data: The input data that contains the target and group data.
    :param variable_type: The type of variable that you want to test. Can be either "continuous" or "categorical".
    :param target_column: The column that contains the target data.
    :param group_column: The column that contains the group data.
    :param target_group: The group that you want to compare to the other group.
    """

    def __init__(
        self,
        input_data,
        variable_type=None,
        target_column=None,
        group_column=None,
        target_group=None,
    ):
        self.variable_type = variable_type
        self.target_column = target_column
        self.group_column = group_column
        self.target_group = target_group
        self.data = input_data.dropna(subset=[self.target_column])
        if self.variable_type == "continuous":
            anova_results = deploy_anova_on_two_groups(
                self.data,
                target_column=self.target_column,
                group_column=self.group_column,
                target_group=self.target_group,
            )
            self.p_value = anova_results[1]
            self.f_statistic = anova_results[0]

        elif self.variable_type == "categorical":
            self.p_value = deploy_chi2_on_two_groups(
                self.data,
                target_column=self.target_column,
                group_column=self.group_column,
                target_group=self.target_group,
            )


def deploy_anova_on_two_groups(
    input_table, target_column="", group_column="", target_group=""
):
    """
    Calculate the F-statistic and p-value for the ANOVA test on two groups.
    Pass in the input table, the target column, the group column, and the target group.
    The target group is the group that you want to compare to the other group. The function will return the F-statistic and p-value.
    This function is useful for comparing two groups to determine if they are significantly different relative to the target column, which is typically a continuous variable.
    :param input_table: The input table that contains the data.
    :param target_column: The column that contains the target data.
    :param group_column: The column that contains the group data.
    :param target_group: The group that you want to compare to the other group.
    :return: The F-statistic and p-value.
    """
    group_data = input_table.loc[
        input_table[group_column] == target_group, target_column
    ]
    not_group_data = input_table.loc[
        input_table[group_column] != target_group, target_column
    ]
    f_stat, p_value = f_oneway(group_data, not_group_data)
    return f_stat, p_value


def deploy_chi2_on_two_groups(
    input_table, target_column="", group_column="", target_group=""
):
    """
    Calculate the p-value for the chi-squared test on two groups.
    Pass in the input table, the target column, the group column, and the target group.
    The target group is the group that you want to compare to the other group. The function will return the p-value.
    This function is useful for comparing two groups to determine if they are significantly different relative to the target column, which is typically a categorical variable.
    :param input_table: The input table that contains the data.
    :param target_column: The column that contains the target data.
    :param group_column: The column that contains the group data.
    :param target_group: The group that you want to compare to the other group.
    """
    grouped_vector = []
    for x in input_table[group_column]:
        if x == target_group:
            grouped_vector.append(1)
        else:
            grouped_vector.append(0)

    contingency_table = pd.crosstab(grouped_vector, input_table[target_column])
    chi_scores = chi2_contingency(contingency_table)
    return chi_scores[1]
