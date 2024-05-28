import gseapy
from sparc_multiomics.MOFA_toolset import readable_features_naming
from sparc_multiomics.plotting import cleaner_barplot


def pathway_analysis(
    input_gene_list,
    p_value=0.01,
    number_of_pathways=15,
    save_info=None,
    return_df=False,
    remove_pathway_source=True,
    split_parameters=None,
    color="#16EB96",
):
    """
    Function to perform pathway analysis using the Enrichr API and plot the top pathways
    :input_gene_list: list of strings containing the gene names to be analyzed
    :p_value: float containing the p-value threshold for the pathways
    :number_of_pathways: integer containing the number of pathways to be plotted
    :save_info: string containing the path to save the plot
    :return_df: boolean indicating if the dataframe containing the pathways should be returned
    :title_text: string containing the title of the plot
    :remove_pathway_source: boolean indicating if the pathway source should be removed from the index
    :split_parameters: dictionary containing the parameters to split the pathway names
        - split_by: string containing the character to split the pathway names
        - split_when: list of splitters containing the characters to split the pathway names
    """
    # Simplify the feature names
    if split_parameters is not None:
        input_gene_list = readable_features_naming(
            input_gene_list,
            split_parameters=split_parameters,
        )

    enrich_object = gseapy.enrichr(
        gene_list=input_gene_list,
        gene_sets=["GO_Biological_Process_2023"],
        organism="Human",
    )
    pathways_df = enrich_object.results

    pathways_df = (
        pathways_df[pathways_df["Adjusted P-value"] < p_value]
        .sort_values(by="Combined Score", ascending=False)
        .head(number_of_pathways)
    )
    cleaner_barplot(
        pathways_df,
        input_x="Combined Score",
        input_y="Term",
        save_info=save_info,
        color=color,
    )
    if remove_pathway_source:
        corrected_pathways_names = []
        for pathway_name in pathways_df["Term"]:
            split_pathway_name = pathway_name.split()
            if ("R-HSA" in split_pathway_name[-1]) or (
                "(GO:" in split_pathway_name[-1]
            ):
                corrected_name = " ".join(split_pathway_name[0:-1])
            else:
                corrected_name = pathway_name
            if len(corrected_name) > 50:
                corrected_name = corrected_name[:50] + " ..."
            corrected_pathways_names.append(corrected_name)
        pathways_df["Term"] = corrected_pathways_names

    if save_info is not None:
        pathways_df.to_csv(f"results/{save_info}.csv", index=False)

    if return_df:
        return pathways_df
