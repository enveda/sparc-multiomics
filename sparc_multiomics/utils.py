from pyensembl import EnsemblRelease


def get_gene_name(
    chromosome: str,
    bp_position: int,
    release_number: int = 110,
) -> str:
    """
    This function returns the gene name for the given chromosome and position.

    Args:
        chromosome (str): The chromosome.
        position (int): The position.

    Returns:
        str: The gene name.
    """
    ensembl = EnsemblRelease(release=release_number)

    # Get genes in the specified position
    return ensembl.gene_names_at_locus(contig=chromosome, position=bp_position)


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


def custom_identify_up_down_regulated_genes(input_target, input_non_target):
    """
    Compare the median expression of the target and non-target groups and identify the up-regulated and down-regulated genes based on the difference in median expression.
    :param input_target: The target group
    :param input_non_target: The non-target group
    :return: The up-regulated and down-regulated genes
    """
    median_target = input_target.median()
    median_non_target = input_non_target.median()

    difference = median_target - median_non_target
    up_regulated = difference[difference > 0].index
    down_regulated = difference[difference < 0].index
    return list(up_regulated), list(down_regulated)
