import random
from typing import List

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sparc_multiomics.constants import RANDOM_SEED
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from statannotations.Annotator import Annotator
from umap import UMAP

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
plt.rcParams["font.family"] = "DejaVu Sans"


def pannel_plot(
    input_data,
    plot_columns,
    x_variable="",
    y_variable="",
    number_of_plots=16,
    title_text="",
    mode="standard",
    save_path=None,
):
    """
    Plot designed to create a pannel of scatter plots for a given set of categorical variables, which will be colored in the scatter plot. They will be plotted against the x and y variables.
    :input_data: pandas DataFrame containing the data to be plotted, both the x and y variables and the categorical variables
    :plot_columns: list of strings containing the names of the categorical variables to be plotted
    :x_variable: string containing the name of the x variable
    :y_variable: string containing the name of the y variable
    :number_of_plots: integer containing the number of plots to be created
    :title_text: string containing the start of the title of the plot
    :mode: string containing the mode of the plot, either "standard" or "agglomeration", when "agglomeration" is selected, the silhouette score will be calculated for each plot
    :save_path: string containing the path to save to. If None, it will not be saved
    """
    plt.rc("legend", fontsize=5, title_fontsize=5)
    # Change the font to DejaVu Sans to prevent font issues

    plot_square = int(number_of_plots ** (1 / 2))
    fig, axs = plt.subplots(
        plot_square,
        plot_square,
        figsize=(10, 10),
    )
    # Adjust the layout parameters to prevent overlap
    plt.subplots_adjust(hspace=0.5, wspace=1.0)

    variable_count = 0
    for i in range(plot_square):
        for j in range(plot_square):
            if variable_count >= len(plot_columns):
                axs[i, j].axis("off")
                continue

            ax = axs[i, j]

            ax.tick_params(axis="both", labelsize=8)
            current_column = plot_columns[variable_count]

            # Only the rows with non-missing values on the categorical variable are used
            usable_rows = input_data[current_column].notna()
            subset_target = input_data[current_column][usable_rows]
            subset_table = input_data[usable_rows]

            # Create a grouped barplot using Seaborn
            plot = sns.scatterplot(
                x=subset_table[x_variable],
                y=subset_table[y_variable],
                hue=subset_target,
                ax=axs[i, j],
                s=8,
            )
            ax.tick_params(axis="both", labelsize=6)
            if len(subset_target.unique()) > 1 and mode == "agglomeration":
                current_silhouette_score = silhouette_score(
                    subset_table[[x_variable, y_variable]].values,
                    subset_target,
                    metric="euclidean",
                )

                rounded_silhouette_score = str(round(current_silhouette_score, 2))
                ax.set_title(
                    f"{current_column}\nSilhouette score:{rounded_silhouette_score}",
                    fontsize=10,
                )
            else:
                ax.set_title(f"{current_column}", fontsize=6)
            ax.set_xlabel(x_variable)
            # Rotate x-labels
            plt.setp(plot.get_xticklabels(), rotation=15)
            ax.set_ylabel(y_variable)
            variable_count += 1

    plt.suptitle(
        f"{title_text} - {x_variable} vs {y_variable} \n (n={input_data.shape[0]})",
        fontsize=12,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()


def pathway_analysis_plot(
    pathways_df, number_of_pathways=15, save_info=None, title_text=""
):
    """
    Draw a barplot of the top pathways
    :pathways_df: pandas DataFrame containing the pathways to be plotted
    :number_of_pathways: integer containing the number of pathways to be plotted
    :save_info: string containing the path to save the plot
    :title_text: string containing the title of the plot
    """
    sns.barplot(
        x="Combined Score",
        y="Term",
        data=pathways_df,
        orient="h",
        color="#16EB96",
    )
    plt.title(f"Top {str(number_of_pathways)} pathways for: {title_text}")

    if save_info is not None:
        plt.savefig(f"figures/{save_info}.png", dpi=600, bbox_inches="tight")
    plt.clf()


def cleaner_barplot(
    input_df,
    input_x="",
    input_y="",
    save_info=None,
    title=None,
    title_fontsize=15,
    color="#16EB96",
):
    """
    Generate a horizontal barplot with the given x and y variables
    :input_df: pandas DataFrame containing the data to be plotted
    :input_x: string containing the name of the x variable
    :input_y: string containing the name of the y variable
    :save_info: string containing the path to save the plot, if None the plot will be displayed
    """
    sns.barplot(
        x=input_x,
        y=input_y,
        data=input_df,
        orient="h",
        color=color,
    )
    plt.xlabel(prettify_text(input_x))
    plt.ylabel(prettify_text(input_y))
    if title:
        plt.title(title, fontsize=title_fontsize)
    else:
        plt.title(f"{prettify_text(input_y)} by {prettify_text(input_x)}")
    if save_info is not None:
        plt.savefig(f"figures/{save_info}.png", dpi=600, bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


def custom_clustermap(
    input_data,
    categories_to_display=List[str],
    columns_to_cluster=List[str],
    save_path=None,
    factor_to_center=None,
    center_on_0=False,
    figsize=(10, 10),
    minimum_columns=5,
    custom_colormap=None,
):
    """
    Generate a custom clustermap with the given categories to display, these will be colored in the rows of the clustermap
    :input_data: pandas DataFrame containing the data to be plotted
    :categories_to_display: list of strings containing the names of the columns to be displayed
    :columns_to_cluster: list of strings containing the names of the columns to be clustered
    :save_path: string containing the path to save the plot, if None the plot will be displayed
    :factor_to_center: string containing the name of the column to be centered on 0
    :center_on_0: boolean indicating if the plot should be centered on 0
    :figsize: tuple containing the size of the figure
    :minimum_columns: integer containing the minimum number of columns to be displayed, if the number of columns is less than this number, empty columns will be added to pad the plot
    :custom_colormap: string containing the name of the custom colormap to be used, if None, the default colormap will be used
    """

    unique_dictionary = {}
    colormaps_dictionary = {}
    colors_dictionary = {}
    mapped_colors = {}
    palette_set_index = 2
    for category in categories_to_display:
        unique_dictionary[category] = input_data[category].unique()
        if len(input_data[category].unique()) > 5:
            colormaps_dictionary[category] = sns.color_palette(
                "tab10", len(input_data[category].unique())
            )

        else:
            colormaps_dictionary[category] = sns.color_palette(
                f"Set{str(palette_set_index)}", len(input_data[category].unique())
            )
            palette_set_index += 1
            if palette_set_index > 3:
                palette_set_index = 1
        colors_dictionary[category] = {
            current_category: colormaps_dictionary[category][i]
            for i, current_category in enumerate(input_data[category].unique())
        }
        mapped_colors[category] = input_data[category].map(colors_dictionary[category])
    max_unique = max(
        [len(unique_dictionary[category]) for category in categories_to_display]
    )

    row_colors_combined = pd.DataFrame(
        {
            prettify_text(category): mapped_colors[category]
            for category in categories_to_display
        }
    )

    if center_on_0 and factor_to_center:
        vmin, vmax = (
            input_data[factor_to_center].min(),
            input_data[factor_to_center].max(),
        )
        normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        colormap = cm.RdBu
    else:
        colormap = "vlag"
        normalize = None

    if custom_colormap is not None:
        colormap = custom_colormap
    data_to_plot = input_data[columns_to_cluster]
    if len(columns_to_cluster) < minimum_columns:
        empty_columns = minimum_columns - len(columns_to_cluster)
        empty_df = pd.DataFrame(
            np.zeros((data_to_plot.shape[0], empty_columns)),
            columns=["" for i in range(empty_columns)],
        )
        data_to_plot = pd.concat([data_to_plot, empty_df], axis=1)

    # Create the clustermap
    current_plot = sns.clustermap(
        data_to_plot,
        col_cluster=False,
        figsize=figsize,
        row_colors=row_colors_combined,
        method="ward",
        metric="euclidean",
        norm=normalize,
        cmap=colormap,
    )

    current_plot.ax_heatmap.set_yticklabels([])
    current_plot.ax_heatmap.tick_params(left=False, bottom=False, right=False)
    all_handles = []
    for category in categories_to_display:
        current_patch = [
            mpatches.Patch(
                color=colors_dictionary[category][category_entry],
                label=f"{prettify_text(category)}: {prettify_text(category_entry)}",
            )
            for category_entry in unique_dictionary[category]
        ]
        while len(current_patch) < max_unique:
            current_patch.append(mpatches.Patch(color="white", label=""))
        all_handles += current_patch
    all_labels = [patch.get_label() for patch in all_handles]

    plt.legend(
        handles=all_handles,
        labels=all_labels,
        loc="center left",
        bbox_to_anchor=(2.0, 0.9),
        ncol=len(categories_to_display),
        prop={"size": 7},
    )

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.clf()


class TwoDimensionalVizualization:
    """
    Class to create a two dimensional vizualization of a dataset using UMAP, t-SNE and PCA
    :data: pandas DataFrame containing the data to be vizualized
    :columns_to_reduce: list of strings containing the names of the columns to be reduced
    :color_variable: string containing the name of the column to be used as color variable
    :save_path: string containing the path to save the plots
    :project_name: string containing the name of the project
    To use the class, first create an instance of the class, then run the plot_all method
    """

    def __init__(
        self,
        data,
        columns_to_reduce=List[str],
        color_variable=str,
        save_path=None,
        project_name=None,
    ):
        self.data = data
        self.columns_to_reduce = columns_to_reduce
        self.color_variable = color_variable
        self.custom_palette = sns.color_palette(["#16EB96", "#F4B183", "#3B3838"])
        self.save_path = save_path
        self.project_name = project_name
        self.title_name = self.project_name.replace("_", " ")

    def run_umap(self):
        UMAP_factors = UMAP(
            n_components=2, transform_seed=RANDOM_SEED, random_state=RANDOM_SEED
        ).fit_transform(self.data[self.columns_to_reduce])
        self.UMAP_factors_df = pd.DataFrame(UMAP_factors, columns=["UMAP 1", "UMAP 2"])

    def run_tsne(self):
        tsne_factors = TSNE(n_components=2).fit_transform(
            self.data[self.columns_to_reduce]
        )
        self.tsne_factors_df = pd.DataFrame(
            tsne_factors, columns=["t-SNE 1", "t-SNE 2"]
        )

    def run_pca(self):
        pca_factors = PCA(n_components=2).fit_transform(
            self.data[self.columns_to_reduce]
        )
        self.pca_factors_df = pd.DataFrame(pca_factors, columns=["PCA 1", "PCA 2"])

    def plot_umap(self):
        sns.scatterplot(
            x=self.UMAP_factors_df["UMAP 1"],
            y=self.UMAP_factors_df["UMAP 2"],
            hue=self.data[self.color_variable],
            palette=self.custom_palette,
        )
        plt.title(f"UMAP of {self.title_name}")
        if self.save_path is not None:
            plt.savefig(f"{self.save_path}/{self.project_name}_UMAP_factors.png")
        plt.show()
        plt.clf()

    def plot_tsne(self):
        sns.scatterplot(
            x=self.tsne_factors_df["t-SNE 1"],
            y=self.tsne_factors_df["t-SNE 2"],
            hue=self.data[self.color_variable],
            palette=self.custom_palette,
        )
        plt.title(f"t-SNE of {self.title_name}")
        if self.save_path is not None:
            plt.savefig(f"{self.save_path}/{self.project_name}_tSNE_factors.png")
        plt.show()
        plt.clf()

    def plot_pca(self):
        sns.scatterplot(
            x=self.pca_factors_df["PCA 1"],
            y=self.pca_factors_df["PCA 2"],
            hue=self.data[self.color_variable],
            palette=self.custom_palette,
        )
        plt.title(f"PCA of {self.title_name}")
        if self.save_path is not None:
            plt.savefig(f"{self.save_path}/{self.project_name}_PCA_factors.png")
        plt.show()
        plt.clf()

    def plot_all(self):
        self.run_umap()
        self.run_tsne()
        self.run_pca()
        self.plot_umap()
        self.plot_tsne()
        self.plot_pca()


def prettify_table(input_table):
    """
    Prettify the column and index names of a pandas DataFrame
    :input_table: pandas DataFrame containing the table to be prettified
    :return: pandas DataFrame with the prettified column and index names
    """
    input_table.index = input_table.index.str.replace("_", " ")
    input_table.index = input_table.index.str.title()
    input_table.columns = input_table.columns.str.replace("_", " ")
    input_table.columns = input_table.columns.str.title()
    input_table = input_table.sort_index()
    return input_table


def prettify_text(input_string):
    """
    Prettify a string by replacing underscores with spaces and capitalizing the first letter
    :input_string: string containing the text to be prettified
    :return: string containing the prettified text
    """
    return str(input_string).replace("_", " ").capitalize()


def unique_ids_venn(input_table, grouping_column="", identifier="", title=None):
    """
    Identify the unique identifiers in a table and create a venn diagram
    :input_table: pandas DataFrame containing the table to be analyzed
    :grouping_column: string containing the name of the column to be used for grouping
    :identifier: string containing the name of the column containing the unique identifiers
    :output_name: string containing the name of the output file, if None, the plot will be displayed
    """
    group_overlap = {}
    for current_group in input_table[grouping_column].unique():
        group_overlap[current_group] = set(
            input_table[input_table["cluster"] == current_group][identifier]
        )

    venn3(
        [group_overlap[current_cluster] for current_cluster in group_overlap.keys()],
        set_labels=[
            f"{grouping_column.capitalize()} {current_cluster}"
            for current_cluster in group_overlap.keys()
        ],
    )
    if title is not None:
        plt.title(title.replace("_", " ").capitalize())
        plt.savefig(
            f"figures/{title.lower().replace(' ', '_')}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.show()
    plt.clf()


def custom_boxplot(
    input_data,
    grouping_variable="",
    x_variable="",
    y_variable="",
    fig_size=(6, 6),
    save_tag=False,
    add_stats=False,
    pairs_to_annotate=None,
    alpha=0.8,
    circle_size=3,
    title=None,
    title_fontsize=12,
    legend_title=None,
    legend_spacing=-0.24,
    prettify_y_axis=False,
    grouped=True,
    skip_sorting=False,
    order=None,
):
    """
    Plot a grouped boxplot of the input data, with the x and y variables and the grouping variable as the hue
    :input_data: pandas DataFrame containing the data to be plotted
    :grouping_variable: string containing the name of the column to be used for grouping
    :x_variable: string containing the name of the x variable
    :y_variable: string containing the name of the y variable
    :save_tag: string containing the path to save the plot, if None the plot will be displayed
    :add_stats: boolean indicating if the statistical tests should be added
    :pairs_to_annotate: list of tuples containing the pairs to be annotated
    :alpha: float containing the transparency of the plot
    :circle_size: float containing the size of the circles
    :title: string containing the title of the plot
    :title_fontsize: integer containing the fontsize of the title
    :legend_title: string containing the title of the legend
    :legend_spacing: float containing the spacing of the legend
    :prettify_y_axis: boolean indicating if the y axis should be prettified
    :grouped: boolean indicating if the boxplot should be grouped
    :skip_sorting: boolean indicating if the data should be sorted
    :order: list of strings containing the order of the x variable
    """
    fig = plt.figure(figsize=fig_size)
    # get ax
    ax = fig.add_subplot(111)

    if grouped:
        if not skip_sorting:
            input_data = input_data.sort_values(by=[x_variable, grouping_variable])
        unique_categories = input_data[grouping_variable].unique()
        pretty_grouping_variable = prettify_text(grouping_variable)

        custom_color_palette = sns.color_palette(["#16EB96", "#F4B183"])
        category_color_dict = dict(zip(unique_categories, custom_color_palette))
    else:
        if not skip_sorting:
            input_data = input_data.sort_values(by=[x_variable])
        custom_color_palette = sns.color_palette(["#16EB96"])

    plot_dict = {
        "x": x_variable,
        "y": y_variable,
        "data": input_data,
        "order": order,
    }

    if grouped:
        plot_dict["hue"] = grouping_variable

    sns.swarmplot(
        **plot_dict,
        alpha=alpha,
        dodge=True,
        size=circle_size,
        palette=custom_color_palette,
    )

    sns.boxplot(
        **plot_dict,
        ax=ax,
        **{
            "boxprops": {"facecolor": "none"},
        },
    )

    if add_stats:
        annotator = Annotator(ax, pairs_to_annotate, **plot_dict, verbose=False)
        annotator.configure(test="Mann-Whitney", text_format="star")
        annotator.apply_and_annotate()

    if grouped:
        legend_elements = []
        for current_category in unique_categories:
            legend_elements.append(
                mpatches.Patch(
                    color=category_color_dict[current_category], label=current_category
                )
            )

        plt.legend(
            handles=legend_elements,
            title=pretty_grouping_variable if not legend_title else legend_title,
            # bottom center and 3 cols
            loc="lower center",
            bbox_to_anchor=(0.5, legend_spacing),
            ncol=2,
        )

    plt.xlabel(prettify_text(x_variable))
    plt.ylabel(prettify_text(y_variable) if prettify_y_axis else title)

    ax.set_xticklabels([x.get_text().capitalize() for x in ax.get_xticklabels()])
    ax.set_yticklabels([y.get_text().capitalize() for y in ax.get_yticklabels()])
    if title:
        plt.title(title, fontsize=title_fontsize)

    else:
        if grouped:
            plt.title(
                f"{prettify_text(y_variable)} by {prettify_text(x_variable)} and {pretty_grouping_variable}"
            )
        else:
            plt.title(f"{prettify_text(y_variable)} by {prettify_text(x_variable)}")
    if save_tag:
        if grouped:
            plt.savefig(
                f"figures/{save_tag}_{y_variable}_by_{x_variable}_and_{grouping_variable}.png",
                dpi=600,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"figures/{save_tag}_{y_variable}_by_{x_variable}.png",
                dpi=600,
                bbox_inches="tight",
            )
    else:
        plt.show()
    plt.clf()


def two_groups_overlap_heatmap(
    input_table, column_1="", column_2="", save_tag=None, normalize_row_wise=False
):
    """
    From the input table, create a heatmap that shows the overlap between the groups in two columns.
    :input_table: pandas DataFrame containing the data to be analyzed
    :column_1: string containing the name of the first column to be analyzed
    :column_2: string containing the name of the second column to be analyzed
    :save_tag: string containing the path to save the plot, if None the plot will be displayed
    :normalize_row_wise: boolean indicating if the heatmap should be normalized row wise
    """
    contigency_table = prettify_table(
        pd.crosstab(
            input_table[column_1],
            input_table[column_2],
        )
    )
    if normalize_row_wise:
        raw_contingency_table = contigency_table.copy()
        contigency_table = contigency_table.div(contigency_table.sum(axis=1), axis=0)
    plt.imshow(contigency_table, cmap="Greens", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(contigency_table.columns)), contigency_table.columns)
    plt.xticks(
        range(len(contigency_table.columns)),
        contigency_table.columns,
        rotation=60,
    )
    plt.yticks(range(len(contigency_table.index)), contigency_table.index)
    plt.ylabel(prettify_text(column_1))
    plt.xlabel(prettify_text(column_2).upper())

    # Annotate heatmap
    for i in range(len(contigency_table)):
        for j in range(len(contigency_table.columns)):
            if normalize_row_wise:
                plt.text(
                    j,
                    i,
                    f"{raw_contingency_table.iloc[i, j]}\n({round(contigency_table.iloc[i, j] * 100, 2)}%)",
                    ha="center",
                    va="center",
                    color="black",
                )
            else:
                plt.text(
                    j,
                    i,
                    contigency_table.iloc[i, j],
                    ha="center",
                    va="center",
                    color="black",
                )

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title(
        f"{prettify_text(column_1)} from {prettify_text(column_2).upper()}",
    )
    plt.tight_layout()
    if save_tag is not None:
        plt.savefig(
            f"figures/{save_tag}_{column_1}_by_{column_2}.png",
            dpi=600,
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.clf()
