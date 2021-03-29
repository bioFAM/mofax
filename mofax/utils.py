import pandas as pd
import numpy as np
from typing import Union, List, Optional

#######################
## Utility functions ##
#######################


def calculate_r2(Z, W, Y):
    a = np.nansum((Y - Z.T.dot(W)) ** 2.0)
    b = np.nansum(Y ** 2)
    r2 = (1.0 - a / b) * 100
    return r2


# def load_samples_metadata(model):

#     # Define metadata template
#     self._samples_metadata = pd.DataFrame(
#         [
#             [sample, group]
#             for group, sample_list in model.samples.items()
#             for sample in sample_list
#         ],
#         columns=["sample", "group"],
#     )

#     # Extract metadata from the model if existing
#     if "samples_metadata" in model:
#         if len(list(model["samples_metadata"][model.groups[0]].keys())) > 0:
#             tmp = pd.concat(
#                 [
#                     pd.concat(
#                         [
#                             pd.Series(model["samples_metadata"][g][k])
#                             for k in model["samples_metadata"][g].keys()
#                         ],
#                         axis=1,
#                     )
#                     for g in model.groups
#                 ],
#                 axis=0,
#             )

#             # ????
#             tmp.columns = list(
#                 model["samples_metadata"][model.groups[0]].keys()
#             )

#             # Merge
#             if "group" in tmp.columns:
#                 del tmp["group"]
#             if "sample" in tmp.columns:
#                 del tmp["sample"]

#             self.samples_metadata = pd.concat(
#                 [
#                     self._samples_metadata.reset_index(drop=True),
#                     tmp.reset_index(drop=True),
#                 ],
#                 axis=1,
#             )

#             # Decode objects as UTF-8 strings
#             for column in self.samples_metadata.columns:
#                 if self.samples_metadata[column].dtype == "object":
#                     try:
#                         self.samples_metadata[column] = [
#                             i.decode() for i in self.samples_metadata[column].values
#                         ]
#                     except (UnicodeDecodeError, AttributeError):
#                         pass

#     self._samples_metadata = self._samples_metadata.set_index("sample")
#     return(self._samples_metadata)


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
