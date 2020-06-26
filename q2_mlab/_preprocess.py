import pandas as pd
import biom
from qiime2 import Metadata


def print_datasize(table, metadata):
    biom_table = table.view(biom.Table)
    dataframe = metadata.to_dataframe()
    print("Table Shape: " + str(biom_table.transpose().shape))
    print("Metadata Shape: " + str(dataframe.shape) + "\n")


def clean_metadata(df: pd.DataFrame, target_variable, discrete):
    # metadata categories that are used in this pipeline are assumed to be
    # cleaned upfront. For classification, the only allowed values are 0 and 1.
    # For regression, the allowed values are any real numbers
    # Enforce numeric for continuous variables
    df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")

    subset_df = df.loc[:, [target_variable]]
    subset_df.dropna(axis=0, subset=[target_variable], inplace=True)

    if discrete:
        val_set = set(subset_df[target_variable].unique())
        if not val_set == {1, 0}:
            raise ValueError(
                "For classification, the only allowed values "
                "in the target column are 0 and 1"
            )
    return subset_df


def preprocess(
    ctx,
    table,
    metadata,
    phylogeny,
    sampling_depth,
    min_frequency,
    target_variable,
    discrete,
    with_replacement=False,
    n_jobs=1,
):

    # Define QIIME2 methods to call
    rarefy = ctx.get_action("feature_table", "rarefy")
    filter_min_features = ctx.get_action("feature_table", "filter_features")
    filter_samples = ctx.get_action("feature_table", "filter_samples")
    beta = ctx.get_action("diversity", "beta")
    beta_phylogenetic = ctx.get_action("diversity", "beta_phylogenetic")
    filter_features = ctx.get_action("fragment-insertion", "filter_features")
    results = []

    print("Inital sizes")
    print_datasize(table, metadata)

    # Filter metadata by samples in table
    print("Filtering Metadata by samples in table")
    ids_to_keep = table.view(biom.Table).ids()
    filteredmetadata = metadata.filter_ids(ids_to_keep=ids_to_keep)
    print_datasize(table, filteredmetadata)

    # Filter samples from metadata where NaN in target_variable column
    # Reduce metadata to 1 column mapping of sample-id to target
    print(
        "Filtering samples from Metadata where NaN in target_variable column"
    )
    print("Reducing Metadata to 1 column mapping of sample-id to target")
    df = filteredmetadata.to_dataframe()
    clean_subset_df = clean_metadata(
        df=df, target_variable=target_variable, discrete=discrete
    )
    target_mapping = Metadata(clean_subset_df)
    print_datasize(table, target_mapping)

    # Filter features that do not exist in phylogeny
    print("Filtering features from Table that do not exist in phylogeny")
    phylo_filtered_results = filter_features(table=table, tree=phylogeny)
    phylo_filtered_table = phylo_filtered_results.filtered_table
    print_datasize(phylo_filtered_table, target_mapping)

    # Filter low-abundance features from table
    print(
        f"Filtering low-abundance features (frequency<{min_frequency}) from Table"
    )
    (filtered_table,) = filter_min_features(
        table=phylo_filtered_table, min_frequency=min_frequency
    )
    print_datasize(filtered_table, target_mapping)

    # Rarefy table to sampling_depth
    print(f"Rarefying Table to sampling depth of {sampling_depth}")
    (rarefied_table,) = rarefy(
        table=filtered_table,
        sampling_depth=sampling_depth,
        with_replacement=with_replacement,
    )
    print_datasize(rarefied_table, target_mapping)

    # Filter table by samples in metadata
    print("Filtering Table by samples in Metadata")
    filtered_rarefied_table_results = filter_samples(
        table=rarefied_table, metadata=filteredmetadata
    )
    filtered_rarefied_table = filtered_rarefied_table_results.filtered_table
    print_datasize(filtered_rarefied_table, target_mapping)

    results += filtered_rarefied_table_results

    # Refilter target_mapping by samples in table
    print("Refiltering Metadata by samples in final Table")
    ids_to_keep = filtered_rarefied_table.view(biom.Table).ids()
    target_mapping = target_mapping.filter_ids(ids_to_keep=ids_to_keep)
    print_datasize(filtered_rarefied_table, target_mapping)

    # Some transformations to get data into correct format for artifact
    target_mapping_col = target_mapping.get_column(target_variable)
    target_mapping_series = target_mapping_col.to_series()
    print("Reindexing Metadata to match Sample ID order of Table")
    target_mapping_series = target_mapping_series.reindex(
        index=ids_to_keep, copy=False
    )
    print("Validating Table and Metadata Sample ID agreement...")
    if list(target_mapping_series.index) != list(ids_to_keep):
        print(list(target_mapping_series.index))
        print(ids_to_keep)
        raise ValueError(
            "Table and Metadata Sample IDs do not match in contents and/or order"
        )
    target_mapping_artifact = ctx.make_artifact(
        "SampleData[Target]", target_mapping_series
    )
    results += [target_mapping_artifact]

    # Generate Distance Matrices
    print("Generating Distance Matrices...")
    for metric in ["jaccard", "braycurtis", "jensenshannon", "aitchison"]:
        beta_results = beta(
            table=filtered_rarefied_table, metric=metric, n_jobs=n_jobs
        )
        results += beta_results
    for metric in ["unweighted_unifrac", "weighted_unifrac"]:
        beta_phylo_results = beta_phylogenetic(
            table=filtered_rarefied_table,
            phylogeny=phylogeny,
            metric=metric,
            n_jobs=n_jobs,
        )
        results += beta_phylo_results

    return tuple(results)
