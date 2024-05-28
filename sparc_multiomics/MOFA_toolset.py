from typing import Dict


def prepare_covariates_MOFA(
    input_table,
    sample_id_column_name=None,
    group_column_name="",
    covariates_columns_names=None,
):
    """
    Prepare the covariates for MOFA analysis. The input table should be in the format of a pandas DataFrame
    :param input_table: The input table containing the data
    :param sample_id_column_name: The name of the column containing the sample IDs
    :param group_column_name: The name of the column containing the group
    :param covariates_columns_names: The list of columns containing the covariates
    :return: The covariates in the format required for MOFA analysis

    Additional Information:
    sample_cov: This can be either
            - 	a list of matrices per group
                The dimensions of each matrix must be (samples, covariates)
                The order of list elements and rows in each matrix must match the structure of data
            - 	a character specifying a column present in the samples' metadata. Note that this requires the metadata
                to have the specified column present as well as the samples names as index.
    covariates_names: String or list of strings containing the name(s) of the covariate(s) (optional)
    REALLY IMPORTANT (2 HOURS WORTH OF IMPORTANCE)
    ent.set_smooth_options(scale_cov=False, model_groups=False) # Model groups is set to False because we are not using it, the Default True usually leads to an error because the covariates don't have the same shape accross groups
    """
    if covariates_columns_names is None:
        covariates_columns_names = list(
            input_table.drop(columns=[sample_id_column_name, group_column_name]).columns
        )

    input_table = input_table.rename(
        columns={group_column_name: "group", sample_id_column_name: "sample"}
    )
    usable_table = input_table[
        ["group", "sample"] + covariates_columns_names
    ].sort_values(by=["group"])
    covariates_list = []
    unique_groups = list(usable_table["group"].unique())
    for current_group in unique_groups:
        current_group_df = usable_table[
            usable_table["group"] == current_group
        ].sort_values(by=["sample"])
        current_covariates_df = current_group_df[covariates_columns_names]
        covariates_list.append(current_covariates_df)
    return covariates_list


def prepare_table_MOFA(
    input_table,
    sample_id_column_name=None,
    view_column_name="",
    group_column_name="",
    features_columns_names=None,
):
    """
    Prepare the table for MOFA analysis. The input table should be in the format of a pandas DataFrame
    :param input_table: The input table containing the data
    :param sample_id_column_name: The name of the column containing the sample IDs
    :param view_column_name: The name of the column containing the view
    :param group_column_name: The name of the column containing the group
    :param features_columns_names: The list of columns containing the features

    :return: The table in the format required for MOFA analysis
    """
    if features_columns_names is None:
        features_columns_names = list(
            input_table.drop(columns=[sample_id_column_name, group_column_name]).columns
        )

    input_table = input_table.rename(
        columns={group_column_name: "group", sample_id_column_name: "sample"}
    )
    usable_table = input_table[["group", "sample"] + features_columns_names]
    melted_table = usable_table.melt(
        id_vars=["group", "sample"],
        value_vars=features_columns_names,
        value_name="value",
        var_name="feature",
    )

    melted_table = melted_table.reset_index(drop=True)
    melted_table["view"] = view_column_name
    melted_table["group"] = melted_table["group"].astype(str).str.replace("_", "-")
    melted_table["sample"] = melted_table["sample"].astype(str).str.replace("_", "-")
    melted_table["feature"] = melted_table["feature"].astype(str).str.replace("_", "-")
    return melted_table[["sample", "group", "feature", "value", "view"]]


def grab_top_features_by_MOFA_weights(
    input_data, target_factors, current_view="", top_n=50
):
    """
    Grab the top features based on their weights of the factors
    :param input_data: The input data containing the weights of the factors
    :param target_factors: The factors to consider
    :param current_view: The current view
    :param top_n: The number of top features to grab, default is 50
    """
    view_data = input_data[input_data["view"] == current_view]
    top_features = []
    for factor in target_factors:
        factor_weights = view_data[factor]
        top_features += (
            factor_weights.abs().sort_values(ascending=False).head(top_n).index.tolist()
        )
    return top_features


def select_top_factors(
    input_r2_factors,
    verbose=False,
    r2_thresholds=None,
    return_mode="overlap",
    top_per_omics=None,
):
    """
    Select the top factors based on the R2 values, the factors with the highest absolute R2 values are selected
    :param input_r2_factors: The input table containing the R2 values of the factors
    :param verbose: Whether to print the top factors
    :param r2_thresholds: The R2 thresholds for each view, if None, the threshold is set to 0.25
    :param return_mode: The mode to return the top factors, either overlap or independent, default is overlap, which returns only the factors that are appear in multiple views.
    :param top_per_omics: The number of top factors to select per omics view, default is 2, if None, all factors are selected
    :return: The top factors, and the views where they appear, if return_mode is overlap, otherwise, the top factors per view
    """

    if r2_thresholds is None:
        r2_thresholds = {}
        for current_view in input_r2_factors["View"].unique():
            r2_thresholds[current_view] = 0.25

    top_factors = {}
    for unique_groups in input_r2_factors["Group"].unique():
        group_subset = input_r2_factors[input_r2_factors["Group"] == unique_groups]
        for current_view in group_subset["View"].unique():
            view_subset = group_subset[group_subset["View"] == current_view]
            view_subset = view_subset.loc[
                view_subset["R2"] > r2_thresholds[current_view]
            ]
            view_subset = view_subset.sort_values(by="R2", ascending=False)
            if current_view not in top_factors:
                top_factors[current_view] = []
            if top_per_omics is not None:
                current_top_factors = view_subset["Factor"].tolist()[0:top_per_omics]
            else:
                current_top_factors = view_subset["Factor"].tolist()
            top_factors[current_view] += current_top_factors

    for current_view in top_factors:
        top_factors[current_view] = list(set(top_factors[current_view]))

    if verbose:
        print(f"Top factors: {top_factors}")

    if return_mode == "overlap":
        output_factors, views_where_factors_appear = [], []
        factor_counter = {}
        for current_view in top_factors:
            for current_factor in top_factors[current_view]:
                if current_factor not in factor_counter:
                    factor_counter[current_factor] = 0
                factor_counter[current_factor] += 1
        for current_factor in top_factors[current_view]:
            if factor_counter[current_factor] > 1:
                output_factors.append(current_factor)
                views_where_factors_appear += [
                    current_view
                    for current_view in top_factors
                    if current_factor in top_factors[current_view]
                ]

        return output_factors, views_where_factors_appear

    elif return_mode == "independent":
        return top_factors


def readable_features_naming(input_list, split_parameters=Dict[str, str]):
    """
    Convert the gene names to a more readable format
    :param input_list: The list of genes
    :param split_parameters: The parameters to split the gene names
    :return: The more readable gene names
    """
    split_gene_list = [x.split(split_parameters["split_by"]) for x in input_list]
    corrected_gene_list = []
    for current_gene in split_gene_list:
        stop_writing = False
        current_name = []
        for current_split in current_gene:
            for current_splitter in split_parameters["split_when"]:
                if current_splitter in current_split:
                    stop_writing = True
            if not stop_writing and current_split != "":
                current_name.append(current_split)
        corrected_gene_list.append("-".join(current_name))
    return corrected_gene_list


def reverse_features_naming(input_list, split_parameters=Dict[str, str]):
    """
    for the MOFA analysis, the feature names are split by a certain character, this function reverses the split
    :param input_list: The list of features
    :param split_parameters: The parameters to split the features
    :return: The reversed feature names
    """
    split_gene_list = [x.split(split_parameters["split_by"]) for x in input_list]
    corrected_gene_list = []
    for current_gene in split_gene_list:
        current_name = []
        found_splitter = False
        for current_split in current_gene:
            for current_splitter in split_parameters["split_when"]:
                if current_splitter in current_split:
                    current_name = ["-".join(current_name)]
                    found_splitter = True
            if current_split != "":
                current_name.append(current_split)
        if found_splitter:
            corrected_gene_list.append("_".join(current_name))
        else:
            corrected_gene_list.append("-".join(current_name))
    return corrected_gene_list
