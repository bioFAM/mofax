import pandas as pd
import numpy as np
from typing import Union, List, Optional


#######################
## Loading metadata  ##
#######################

def _load_samples_metadata(model):
    samples_metadata = pd.DataFrame(
        [
            [cell, group]
            for group, cell_list in model.samples.items()
            for cell in cell_list
        ],
        columns=["sample", "group"],
    )
    if "samples_metadata" in model.model:
        if len(list(model.model["samples_metadata"][model.groups[0]].keys())) > 0:
            _samples_metadata = pd.concat(
                [
                    pd.concat(
                        [
                            pd.Series(model.model["samples_metadata"][g][k])
                            for k in model.model["samples_metadata"][g].keys()
                        ],
                        axis=1,
                    )
                    for g in model.groups
                ],
                axis=0,
            )
            _samples_metadata.columns = list(
                model.model["samples_metadata"][model.groups[0]].keys()
            )

            if "group" in _samples_metadata.columns:
                del _samples_metadata["group"]
            if "sample" in _samples_metadata.columns:
                del _samples_metadata["sample"]

            samples_metadata = pd.concat(
                [
                    samples_metadata.reset_index(drop=True),
                    _samples_metadata.reset_index(drop=True),
                ],
                axis=1,
            )

            # Decode objects as UTF-8 strings
            for column in samples_metadata.columns:
                if samples_metadata[column].dtype == "object":
                    try:
                        samples_metadata[column] = [
                            i.decode() for i in samples_metadata[column].values
                        ]
                    except (UnicodeDecodeError, AttributeError):
                        pass

    samples_metadata = samples_metadata.set_index("sample")
    return samples_metadata


def _load_features_metadata(model):
    features_metadata = pd.DataFrame(
        [
            [feature, view]
            for view, feature_list in model.features.items()
            for feature in feature_list
        ],
        columns=["feature", "view"],
    )
    if "features_metadata" in model.model:
        if len(list(model.model["features_metadata"][model.views[0]].keys())) > 0:
            features_metadata_dict = {
                m: pd.concat(
                    [
                        pd.Series(model.model["features_metadata"][m][k])
                        for k in model.model["features_metadata"][m].keys()
                    ],
                    axis=1,
                )
                for m in model.views
            }

            for m in features_metadata_dict.keys():
                features_metadata_dict[m].columns = list(
                    model.model["features_metadata"][m].keys()
                )

            _features_metadata = pd.concat(features_metadata_dict, axis=0)

            if "view" in _features_metadata.columns:
                del _features_metadata["view"]
            if "feature" in _features_metadata.columns:
                del _features_metadata["feature"]

            features_metadata = pd.concat(
                [
                    features_metadata.reset_index(drop=True),
                    _features_metadata.reset_index(drop=True),
                ],
                axis=1,
            )

            # Decode objects as UTF-8 strings
            for column in features_metadata.columns:
                if features_metadata[column].dtype == "object":
                    try:
                        features_metadata[column] = [
                            i.decode()
                            for i in features_metadata[column].values
                        ]
                    except (UnicodeDecodeError, AttributeError):
                        pass

    features_metadata = features_metadata.set_index("feature")
    return features_metadata

def _load_covariates(model):
    samples_covariates = pd.DataFrame(
        [
            [cell, group]
            for group, cell_list in model.samples.items()
            for cell in cell_list
        ],
        columns=["sample", "group"],
    )
    for attr in ["cov_samples", "cov_samples_transformed"]:
        if attr in model.model:
            if len(model.model[attr].keys()) > 0:  # groups are not empty
                attr_covariates = np.concatenate([np.array(model.model[attr][g]) for g in model.groups])
                samples_covariates[attr] = attr_covariates
    
    # Covariates are None when there is only sample name and group name
    if len(samples_covariates.columns) == 2:
        return None

    samples_covariates = samples_covariates.set_index("sample")
    return samples_covariates



#######################
## Utility functions ##
#######################

def calculate_r2(Z, W, Y):
    a = np.nansum((Y - Z.T.dot(W)) ** 2.0)
    b = np.nansum(Y ** 2)
    r2 = (1.0 - a / b) * 100
    return r2

def padjust_fdr(xs):
    """
    Adjust p-values using the BH procedure
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(xs)
    fdr = xs * len(xs) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


def padjust_fdr_2d(mx):
    """
    Adjust p-values in a matrix using the BH procedure
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(mx).reshape((-1, mx.shape[1]))
    fdr = mx * mx.shape[0] * mx.shape[1] / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr
