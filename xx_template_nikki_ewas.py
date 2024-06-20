# -*- coding: utf-8 -*-
"""
Workflow develpment by Nikki to EWAS Analysis with ECHO Data

Clean Code to become a tutorial than a real analysis script
"""

import pandas as pd
import clarite

main_table_updated_pre = pd.DataFrame

# Split main table by variable type and check data types
split_main_table_pre = clarite.modify.categorize(main_table_updated_pre)

main_type_pre = clarite.describe.get_types(split_main_table_pre)

# Designate data types and drop unnescessary columns and 'constant' data types
split_main_table_drop_pre = clarite.modify.colfilter(
    split_main_table_pre,
    skip=['xDateOfBirth', 'specific_gravity'])

# make cat
main_make_cat_pre = clarite.modify.make_categorical(
    split_main_table_drop_pre,
    only=['CohortID', 'cov-demchild_race', 'exp-block3_milktype'])

main_make_cat_cont_pre = clarite.modify.make_continuous(
    main_make_cat_pre,
    only=['exp-calibrated_pre_poll_avg30', 'As']
    )

# Split by cohort to make discovery and replication groups
rep_main_pre = main_make_cat_cont_pre.drop(
    main_make_cat_cont_pre.index[
        main_make_cat_cont_pre['CohortID'].isin([11901, 13301])
        ]).reset_index()
dis_main_pre = main_make_cat_cont_pre.drop(
    main_make_cat_cont_pre.index[
        main_make_cat_cont_pre['CohortID'].isin([13502, 11202])
        ])

# Separate out phenotypes from exposures, make sure age for pre and sch are
# separated becasue of varying NAs
exp_dis_main_pre = clarite.modify.colfilter(
    dis_main_pre,
    skip=['phe-ADHD Pre', 'phe-Autism Pre']
    )
pheno_dis_main_pre = clarite.modify.colfilter(
    dis_main_pre,
    only=['phe-ADHD Pre', 'phe-Autism Pre']
    )

# Only keep complete cases (Probably will have to skip this)
covariates_pre = []
dis_exp_filtered_pre = clarite.modify.rowfilter_incomplete_obs(
    exp_dis_main_pre,
    only=covariates_pre
    )

# PHENOTYPE QC##
# Calculate percent zero for continuous phenotypes
dis_phe_filtered_pre = clarite.modify.colfilter_percent_zero(
    pheno_dis_main_pre
    )

# Get skewness values and histogram plots of the phenotypes
clarite.describe.skewness(
    dis_phe_filtered_pre,
    dropna='True'
    )

# Plot histogram of phenotype and show skewness values
title = f"Discovery: Skew ADHD Age 1.5-5 = {dis_phe_filtered_pre['phe-ADHD Pre'].skew(axis = 0, skipna = True):.6}" # noqa E501
clarite.plot.histogram(
    dis_phe_filtered_pre,
    column="phe-ADHD Pre",
    title=title,
    bins=100
    )

# Log transform all phenotypes since the skewness values are greater than 0.5
dis_pheno_log_transform_pre = clarite.modify.transform(
    dis_phe_filtered_pre,
    'log'
    )

# Run skewness again after log transf
clarite.describe.skewness(
    dis_pheno_log_transform_pre,
    dropna='True'
    )

title = f"Discovery: Skew ADHD Age 1.5-5 = {dis_pheno_log_transform_pre['phe.ADHD.Pre'].skew(axis = 0, skipna = True):.6}" # noqa E501
clarite.plot.histogram(
    dis_pheno_log_transform_pre,
    column="phe.ADHD.Pre",
    title=title,
    bins=100
    )


# QC OF EXPOSURES##

# Filter by requiring a min of 50 and 75 non-null and unique occurences in
# samples after spliting. Remove binary and categorical variables whioch
# have category with less than 10 values.
dis_exp_min_n_pre = clarite.modify.colfilter_min_n(
    dis_exp_filtered_pre,
    n=50,
    skip=covariates_pre
    )

dis_exp_min_cat_n_pre = clarite.modify.colfilter_min_cat_n(
    dis_exp_min_n_pre,
    n=10,
    skip=covariates_pre
    )

# Only keep varaibles that can be found in both discovery tables
# I ran PheEWAS for these in sets for discovery and replication per age group
rep_exp_min_cat_n_pre = pd.DataFrame
list_df = [dis_exp_min_cat_n_pre, rep_exp_min_cat_n_pre]
pre_col_common = set.intersection(*(set(df.columns) for df in list_df))
dis_pre_common_cols = clarite.modify.colfilter(
    dis_exp_min_cat_n_pre,
    only=pre_col_common
    )

# merge phenotype variables back with exposures
disc_pre_phe = dis_pre_common_cols.merge(
    dis_phe_filtered_pre,
    on='ID',
    how='left'
    )

# Isolate the cluster column xFamilyID (RUN FIRST)
Family_df = clarite.modify.colfilter(
    disc_pre_phe,
    only=['xFamilyID']
    )
disc_pre_phe_no_fid = clarite.modify.colfilter(
    disc_pre_phe,
    skip=['xFamilyID']
    )

# Get survey design for discovery and replication datasets
survey_design_discovery_match = Family_df.loc[disc_pre_phe_no_fid.index]

# Create for design CRSE (You will run this for each age group for
# fisrt and second run of PheEWAS)
design_dis = clarite.survey.SurveyDesignSpec(
    survey_design_discovery_match,
    weights=None,
    cluster="xFamilyID",
    strata=None,
    drop_unweighted=False,
    fpc=None,
    nest=False
    )

# STEP 1: Isolate the variables in a list
phenotypes_pre = []
list_skip_columns = covariates_pre + phenotypes_pre
variables = list(
    clarite.modify.colfilter(
        disc_pre_phe_no_fid,
        skip=list_skip_columns
        ).columns
        )

# STEP 2: Run variables one-by-one
result = []
for outcome in phenotypes_pre:

    for variable in variables:
        # STEP 2.1: Isolate only fields to process in a copy of the datas
        v_list = covariates_pre + [outcome] + [variable]
        data = disc_pre_phe_no_fid.copy()
        data = clarite.modify.colfilter(data, only=v_list)
        # STEP 2.2: Drop all row with any NAN
        data = data.dropna()
        # STEP 2.3: Run Regression
        disc_pre_pheewas = clarite.analyze.association_study(
            data=data,
            outcomes=outcome,
            covariates=covariates_pre,
            survey_design_spec=design_dis,
            min_n=10,
            standardize_data=False,
            process_num=4
            )
        result.append(disc_pre_pheewas)

# STEP 3: consolidates all regressions into a single dataframe
disc_pre_pheewas = pd.concat(result)
clarite.analyze.add_corrected_pvalues(disc_pre_pheewas)

# Get a dictionary of phenotype : list of significant variables
significant_results_disc_pre = disc_pre_pheewas[
    disc_pre_pheewas['pvalue_fdr'] < 0.1
    ].index.values
sr_dict = dict()
for var, phenotype in significant_results_disc_pre:
    if phenotype in sr_dict:
        sr_dict[phenotype].append(var)
    else:
        sr_dict[phenotype] = [var]
