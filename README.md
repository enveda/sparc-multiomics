# SPARC Multiomics

This repository details the steps needed to reproduce the analysis described in the publication:

## Table of Contents

* [Citation](#citation)
* [Setup](#setup)
* [Reproducibility](#reproducibility)
  * [Metadata](#metadata)
  * [Transcriptomics](#transcriptomics)
  * [Genomics](#genomics)
  * [Proteomics](#proteomics)
  * [MOFA](#mofa)
  * [Predictions](#predictions)
* [References](#references)
* [Acknowledgments](#acknowledgments)

## Citation

**Leveraging multi-omics data for precision medicine and biomarker discovery in inflammatory bowel disease**, *António José Preto*, *Shaurya Chanana*, *Daniel Ence*, *Daniel Domingo-Fernández*, *Kiana West*, 2024

## Setup
When replicating the project, we are unable to disclose the real data, as it was provided by the [Chron's and Colitis Foundation](https://www.crohnscolitisfoundation.org/) and is not open source. In such cases, we will indicate in the notebook and proceed with the code considering the starting and end points of the process.
To install the needed dependencies, use [poetry](https://python-poetry.org/) and run, inside the folder:
`poetry install`

## Reproducibility

We will display the steps needed to recreate the experiments we performed in our protocol, for such, we will go through the different categories and, in there, through the ordered sequence of notebooks. This can be followed through in the `notebook` folder.

### Metadata

The subsection pertains the `notebook/metadata` folder:
1. Firstly, we performed a semi-automated protocol to curate the metadata into manageable format

    * *folder: `metadata`*
    * *notebook: `1-metadata_processing.ipynb`*
    * *input_file: raw_metadata.xlsx*
    * *output_file: processed_metadata.parquet*

2. Then, we aggregated the omics samples by their collection data, into a collapsed format

    * *folder: `metadata`*
    * *notebook: `2-metadata_collection_date_aggregation.ipynb`*
    * *input_file: processed_metadata.parquet*
    * *output_file: collapsed_metadata.parquet*

### Transcriptomics

This subsection pertains the `notebook/transcriptomics` folder.

3. Starting from the initial transcriptomics vendor data, we performed initial processing, namely normalized using `pydeseq2`, multiply by the size factors and dropping features with variance 0 and rearranging the columns to later be able to merge with the metadata.

    * *folder: `transcriptomics`*
    * *notebook: `3-transcriptomics_raw_processing.ipynb`*
    * *input_file: "transcriptomics_raw_data.txt.gz"* 
    * *output_file: "transcriptomics_pydeseq_corrected.parquet"*

4. Use the `batch` information from the metadata table to perform batch correction. We deploy our own batch correction step based on average intra-batch variance and subsequently proceeded with the standard `pycombat` approach.

    * *folder: `transcriptomics`*
    * *notebook: `4-transcriptomics_batch_effect_correction.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_pydeseq_corrected.parquet"* 
    * *output_file: "transcriptomics_batch_corrected.parquet"*

### Genomics

5. Processing and annotating the genomics data.
- A. From the original genomics array vendor data, we first deploy a standalone version of [vcftools](https://vcftools.github.io/man_latest.html), with which we process the raw data to reduce it to biallelic sites with a minor allele frequency at least 0.05. The command used is:

    - `nohup vcftools --gzvcf data/anvil_ccdg_broad_ai_ibd_daly_lewis_sparc_gsa.vcf.gz --recode --min-alleles 2 --max-alleles 2 --maf 0.05 --recode-INFO-all --remove-filtered-all &> vcftools.out & `

- B. With the data processed through the standalone vcftools software, we use `cyvcf2` to get the data into a readable format.

    * *folder: `genomics`*
    * *notebook: `5-genomics_raw_to_processed.ipynb`*
    * *input_file: "out.recode.vcf"* 
    * *output_file: "genomics_processed.parquet"*

6. Annotating the genomics data using `pyensembl`.

    * *folder: `genomics`*
    * *notebook: `6-genomics_annotation.ipynb`*
    * *input_file: "genomics_processed.parquet"* 
    * *output_file: "genomics_annotated.parquet"*

### Proteomics

Proteomics Olink data did not undergo much processing, aside from column names changing and exclusion of samples (those with dubious quality).

7. Changes over the vendor data.

    * *folder: `proteomics`*
    * *notebook: `7-proteomics_data_cleanup.ipynb`*
    * *input_file: "20211937_Dobes_NPX_2022-06-07.csv"* 
    * *output_file: "proteomics_processed.parquet"*

### MOFA

We ran [MOFA](https://github.com/bioFAM/mofapy2) on our data as a way to perform data integration. Ultimately, we look at the factors generated in order to attempt to associated with phenotypical and biochemical profiles.

8. Running MOFA with `mofapy` with `macroscopic_appearance` as the `group` variable
- A. ulcerative colitis in the colon tissue

    * *folder: `MOFA`*
    * *notebook: `8A-MOFA_uc_colon.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_batch_corrected.parquet", "genomics_annotated.parquet", "proteomics_processed.parquet"* 
    * *output_file: "mofa_inflammation_uc_colon.h5"*

- B. Chron's disease in the colon tissue

    * *folder: `MOFA`*
    * *notebook: `8B-MOFA_cd_colon.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_batch_corrected.parquet", "genomics_annotated.parquet", "proteomics_processed.parquet"* 
    * *output_file: "mofa_inflammation_cd_colon.h5"*

9. Analyzing MOFA with pathway analysis

Perform clustering on the MOFA factors and then use a combination of mutual information, chi-square and pathway analysis to link the clusters to feature information.
- A. ulcerative colitis in the colon tissue
    * *folder: `MOFA`*
    * *notebook: `9A-MOFA_results_analysis_uc_colon.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_batch_corrected.parquet", "genomics_annotated.parquet", "proteomics_processed.parquet", "mofa_inflammation_uc_colon.h5"* 
    * *output_file: this notebook outputs a lot of files, which require a `figures` and a results `folder`, it also outputs a file needed for the next steps "MOFA_results_uc_colon.parquet"*

- B. Chron's disease in the colon tissue
    * *folder: `MOFA`*
    * *notebook: `9B-MOFA_results_analysis_cd_colon.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_batch_corrected.parquet", "genomics_annotated.parquet", "proteomics_processed.parquet", "mofa_inflammation_cd_colon.h5"* 
    * *output_file: this notebook outputs a lot of files, which require a `figures` and a results `folder`, it also outputs a file needed for the next steps "MOFA_results_cd_colon.parquet"*

10. Analyzing MOFA clustering

Analyze the MOFA clustering with 2D visualization.
- A. ulcerative colitis in the colon tissue
    * *folder: `MOFA`*
    * *notebook: `10A-MOFA_cluster_analysis_uc_colon.ipynb`*
    * *input_files: "MOFA_results_uc_colon.parquet"* 
    * *output_file: images to the `figures` folder*

- B. Chron's disease in the colon tissue
    * *folder: `MOFA`*
    * *notebook: `10B-MOFA_cluster_analysis_cd_colon.ipynb`*
    * *input_files: "MOFA_results_cd_colon.parquet"* 
    * *output_file: images to the `figures` folder*

### Predictions

11. UC/CD prediction

- Train a prediction model using Extreme Gradient Boosting to predict between UC/CD diagnosis.
    * *folder: `UC-CD-prediction`*
    * *notebook: `11-multix_supervised_learning_uc_cd.ipynb`*
    * *input_files: "collapsed_metadata.parquet", "transcriptomics_batch_corrected.parquet", "genomics_annotated.parquet", "proteomics_processed.parquet"* 
    * *output_files: "results/uc_cd_predictions.csv", "results/features_importance.csv"*

## References

This references were particularly relevant for this work.

- *Argelaguet, R., Velten, B., Arnol, D., Dietrich, S., Zenz, T., Marioni, J. C., et al.* (2018). **Multi‐Omics Factor Analysis—a framework for unsupervised integration of multi‐omics data sets.** Molecular systems biology, 14(6), e8124. [Paper](https://doi.org/10.15252/msb.20178124), [Repository](https://github.com/bioFAM/mofapy2) 
- *Behdenna, A., Colange, M., Haziza, J., Gema, A., Appé, G., Azencott, C. A., and Nordor, A.* (2023). **pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods.** BMC bioinformatics, 24(1), 459. [Paper](https://doi.org/10.1186/s12859-023-05578-5), [Repository](https://github.com/epigenelabs/inmoose)
  
## Acknowledgments
The results published here are partly based on data from the Study of a Prospective Adult Research Cohort with IBD (SPARC IBD). SPARC IBD is a component of the Crohn’s & Colitis Foundation’s IBD Plexus data exchange platform. SPARC IBD enrolls patients with an established or new diagnosis of IBD from sites throughout the United States and links data collected from the electronic health record and study specific case report forms. Patients also provide blood, stool, and biopsy samples at selected times during follow-up. The design and implementation of the SPARC IBD cohort has been previously described.
