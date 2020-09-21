import h5py
import numpy as np
import pandas as pd

import sys
from os import path
from typing import Union, List, Optional
from collections.abc import Iterable
import warnings


class mofa_model:
    """Class around HDF5-based model on disk.

    This class is a thin wrapper for the HDF5 file where the trained MOFA+ model is stored.
    It also provides utility functions to get factors, weights, features, and samples (cells) info
    in the form of Pandas dataframes, and data as a NumPy array.
    """

    def __init__(self, filepath, mode="r"):
        self.filepath = filepath
        self.filename = path.basename(filepath)
        self.model = h5py.File(filepath, mode)

        self.data = self.model["data"] if 'data' in self.model else None

        self.samples = {
            g: np.array(self.model["samples"][g]).astype("str")
            for g in self.model["samples"]
        }
        self.features = {
            m: np.array(self.model["features"][m]).astype("str")
            for m in self.model["features"]
        }

        self.groups = list(self.model["samples"].keys())
        self.views = list(self.model["features"].keys())

        self.expectations = self.model["expectations"]
        self.factors = self.model["expectations"]["Z"]
        self.weights = self.model["expectations"]["W"]

        if self.data is not None:
            self.shape = (
                sum(self.data[self.views[0]][group].shape[0] for group in self.groups),
                sum(self.data[view][self.groups[0]].shape[1] for view in self.views),
            )
        else:
            self.shape = (
                sum(self.factors[group].shape[0] for group in self.groups),
                sum(self.weights[view].shape[0] for view in self.views)
            )
        self.nfactors = self.model["expectations"]["Z"][self.groups[0]].shape[0]
        self.nviews = len(self.views)
        self.ngroups = len(self.groups)

        if "model_options" in self.model:
            self.likelihoods = (
                np.array(self.model["model_options"]["likelihoods"]).astype("str").tolist()
            )

        if "training_opts" in self.model:
            # TODO: Update according to the latest API
            self.training_opts = {"maxiter": self.model["training_opts"][0]}


        self._samples_metadata = pd.DataFrame(
            [
                [cell, group]
                for group, cell_list in self.samples.items()
                for cell in cell_list
            ],
            columns=["sample", "group"],
        )
        if 'samples_metadata' in self.model:
            if len(list(self.model['samples_metadata'][self.groups[0]].keys())) > 0:
                samples_metadata = pd.concat(
                    [
                        pd.concat([pd.Series(self.model['samples_metadata'][g][k]) for k in self.model['samples_metadata'][g].keys()], axis=1)
                        for g in self.groups
                    ],
                    axis=0
                )
                samples_metadata.columns = list(self.model['samples_metadata'][self.groups[0]].keys())

                self.samples_metadata = pd.concat([self._samples_metadata, samples_metadata], axis=1)

                # Decode objects as UTF-8 strings
                for column in self.samples_metadata.columns:
                    if self.samples_metadata[column].dtype == "object":
                        try:
                            self.samples_metadata[column] = [i.decode() for i in self.samples_metadata[column].values]
                        except (UnicodeDecodeError, AttributeError):
                            pass
                        

        self._samples_metadata = self._samples_metadata.set_index("sample")
        

        self.features_metadata = pd.DataFrame(
            [
                [feature, view]
                for view, feature_list in self.features.items()
                for feature in feature_list
            ],
            columns=["feature", "view"],
        )
        if 'features_metadata' in self.model:
            if len(list(self.model['features_metadata'][self.views[0]].keys())) > 0:
                features_metadata_dict = {
                        m:pd.concat([pd.Series(self.model['features_metadata'][m][k]) for k in self.model['features_metadata'][m].keys()], axis=1)
                        for m in self.views
                    }

                for m in features_metadata_dict.keys():
                    features_metadata_dict[m].columns = list(self.model['features_metadata'][m].keys())
                
                features_metadata = pd.concat(features_metadata_dict, axis=0)
                
                self.features_metadata = pd.concat([self._features_metadata, features_metadata.reset_index()], axis=1)

                # Decode objects as UTF-8 strings
                for column in self.features_metadata.columns:
                    if self.features_metadata[column].dtype == "object":
                        try:
                            self.features_metadata[column] = [i.decode() for i in self.features_metadata[column].values]
                        except (UnicodeDecodeError, AttributeError):
                            pass

        self.features_metadata = self.features_metadata.set_index("feature")
        

    def __repr__(self):
        return(f"""MOFA+ model: {" ".join(self.filename.replace(".hdf5", "").split("_"))}
Samples (cells): {self.shape[0]}
Features: {self.shape[1]}
Groups: {', '.join([f"{k} ({len(v)})" for k, v in self.samples.items()])}
Views: {', '.join([f"{k} ({len(v)})" for k, v in self.features.items()])}""")

    # Alias samples as cells
    @property
    def cells(self):
        return self.samples

    @property
    def samples_metadata(self):
        return self._samples_metadata

    @samples_metadata.setter
    def samples_metadata(self, metadata):
        if len(metadata) != self.shape[0]:
            raise ValueError(f"Length of provided metadata {len(metadata)} does not match the length {self.shape[0]} of the data.")
        self._samples_metadata = metadata

    @property
    def cells_metadata(self):
        return self.samples_metadata

    @cells_metadata.setter
    def cells_metadata(self, metadata):
        self.samples_metadata = metadata
    
    @property
    def metadata(self):
        return self.samples_metadata

    @metadata.setter
    def metadata(self, metadata):
        self.samples_metadata = metadata

    @property
    def features_metadata(self):
        return self._features_metadata

    @features_metadata.setter
    def features_metadata(self, metadata):
        if len(metadata) != self.shape[1]:
            raise ValueError(f"Length of provided metadata {len(metadata)} does not match the length {self.shape[1]} of the data.")
        self._features_metadata = metadata

    def close(self):
        """Close the connection to the HDF5 file"""
        if self.model.__bool__():  # if the connection is still open
            self.model.close()

    def get_shape(self, groups=None, views=None):
        """
        Get the shape of all the data, samples (cells) and features pulled across groups and views.

        Parameters
        ----------
        groups : optional
            List of groups to consider
        views : optional
            List of views to consider
        """
        groups = self.__check_groups(groups)
        views = self.__check_views(views)
        shape = (
            sum(self.data[self.views[0]][group].shape[0] for group in groups),
            sum(self.data[view][self.groups[0]].shape[1] for view in views),
        )
        return shape

    def get_samples(self, groups=None):
        """
        Get the sample metadata table (sample ID and its respective group)

        Parameters
        ----------
        groups : optional
            List of groups to consider
        """
        groups = self.__check_groups(groups)
        return pd.DataFrame(
            [
                [group, cell]
                for group, cell_list in self.cells.items()
                for cell in cell_list
                if group in groups
            ],
            columns=["group", "sample"],
        )

    # Alias samples as cells
    def get_cells(self, groups=None):
        """
        Get the cell metadata table (cell ID and its respective group)

        Parameters
        ----------
        groups : optional
            List of groups to consider
        """
        cells = self.get_samples(groups)
        cells.columns = ["group", "cell"]
        return cells


    def get_features(self, views=None):
        """
        Get the features metadata table (feature name and its respective view)

        Parameters
        ----------
        views : optional
            List of views to consider
        """
        views = self.__check_views(views)
        return pd.DataFrame(
            [
                [view, feature]
                for view, feature_list in self.features.items()
                for feature in feature_list
                if view in views
            ],
            columns=["view", "feature"],
        )

    def get_top_features(
        self,
        factors: Union[int, List[int]] = None,
        views: Union[str, int, List[str], List[int]] = None,
        n_features: int = None,
        clip_threshold: float = None,
        scaled_values: bool = False,
        absolute_values: bool = False,
        only_positive: bool = False,
        only_negative: bool = False
    ):
        """
        Fetch a list of top feature names

        Parameters
        ----------
        factors : optional
            Factors to use (all factors in the model by default)
        view : options
            The view to get the factor weights for (first view by default)
        n_features : optional
            Number of features for each factor by their absolute value (10 by default)
        clip_threshold : optional
            Absolute weight threshold to clip all values to (no threshold by default)
        absolute_values : optional
            If to fetch absolute weight values
        only_positive : optional
            If to fetch only positive weights
        only_negative : optional
            If to fetch only negative weights
        """
        views = self.__check_views(views)
        findices, factors = self.__check_factors(factors)
        n_features_default = 10
        
        # Fetch weights for the relevant factors
        w = (
            self.get_weights(views=views, factors=factors, df=True, absolute_values=absolute_values)
            .rename_axis("feature")
            .reset_index()
        )
        wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
        wm = wm.assign(value_abs=lambda x: x.value.abs())
        wm["factor"] = wm["factor"].astype("category")

        if only_positive and only_negative:
            print("Please specify either only_positive or only_negative")
            sys.exit(1)
        elif only_positive:
            wm = wm[wm.value > 0]
        elif only_negative:
            wm = wm[wm.value < 0]

        if n_features is None and clip_threshold is not None:
            features = wm[wm.value_abs >= clip_threshold].feature.unique()
        else:
            if n_features is None:
                n_features = n_features_default
            # Get a subset of features
            wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby("factor")
            if clip_threshold is None:
                features = wm.head(n_features).feature.unique()
            else:
                features = wm[wm.value_abs >= clip_threshold].head(n_features).feature.unique()

        return features

    def get_factors(
        self,
        groups: Union[str, int, List[str], List[int]] = None,
        factors: Union[int, List[int]] = None,
        df=False,
        scaled_values: bool = False,
        absolute_values: bool = False,
    ):
        """
        Get the matrix with factors as a NumPy array or as a DataFrame (df=True).

        Parameters
        ----------
        groups : optional
            List of groups to consider
        factors : optional
            Indices of factors to consider
        df : optional
            Boolean value if to return Z matrix as a DataFrame
        scaled_values : optional
            If return values scaled to zero mean and unit variance (per sample or cell)
        absolute_values : optional
            If return absolute values for factors
        """
        groups = self.__check_groups(groups)
        findices, factors = self.__check_factors(factors)
        z = np.concatenate(
            tuple(np.array(self.factors[group]).T[:, findices] for group in groups)
        )
        if scaled_values:
            z = (z - z.mean(axis=0)) / z.std(axis=0)
        if absolute_values:
            z = np.absolute(z)
        if df:
            z = pd.DataFrame(z)
            z.columns = factors
            z.index = np.concatenate(tuple(self.samples[g] for g in groups))
        return z

    def get_weights(
        self,
        views: Union[str, int, List[str], List[int]] = None,
        factors: Union[int, List[int]] = None,
        df: bool = False,
        scaled_values: bool = False,
        absolute_values: bool = False,
    ):
        """
        Get the matrix with weights as a NumPy array or as a DataFrame (df=True).

        Parameters
        ----------
        views : optional
            List of views to consider
        factors : optional
            Indices of factors to use
        df : optional
            Boolean value if to return W matrix as a DataFrame
        scaled_values : optional
            If return values scaled to zero mean and unit variance (per gene)
        absolute_values : optional
            If return absolute values for weights
        """
        views = self.__check_views(views)
        findices, factors = self.__check_factors(factors)
        w = np.concatenate(
            tuple(np.array(self.weights[view]).T[:, findices] for view in views)
        )
        if scaled_values:
            w = (w - w.mean(axis=0)) / w.std(axis=0)
        if absolute_values:
            w = np.absolute(w)
        if df:
            w = pd.DataFrame(w)
            w.columns = factors
            w.index = np.concatenate(tuple(self.features[m] for m in views))
        return w

    def get_data(
        self,
        features: Optional[Union[str, List[str]]],
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        df=False,
    ):
        """
        Get the subset of the training data matrix as a NumPy array or as a DataFrame (df=True).

        Parameters
        ----------
        groups : optional
            List of groups to consider
        features : optional
            Features to consider (from one view)
        df : optional
            Boolean value if to return Y matrix as a DataFrame
        """
        groups = self.__check_groups(groups)
        # If a sole feature name is used, wrap it in a list
        if not isinstance(features, Iterable) or isinstance(features, str):
            features = [features]
        else:
            # Make feature names unique
            features = list(set(features))

        # Deduce the view from the feature name
        fs = self.get_features()
        f_i = np.where(fs.feature.isin(features))[0]
        assert len(f_i) > 0, "Requested features are not found"
        f_view = fs.iloc[f_i, :].view.unique()
        assert len(f_view) == 1, "All the features should be from one view"
        f_view = f_view[0]

        # Determine feature index in that view
        fs = self.get_features(views=f_view)
        f_i = np.where(fs.feature.isin(features))[0]

        y = np.concatenate(
            tuple(np.array(self.data[f_view][group])[:, f_i] for group in groups)
        )
        if df:
            y = pd.DataFrame(y)
            y.columns = fs.feature.values[f_i]
            y.index = np.concatenate(tuple(self.samples[g] for g in groups))
        return y

    def fetch_values(self, variables: Union[str, List[str]]):
        """
        Fetch metadata column, factors, or feature values.
        Shorthand to get_data, get_factors, and metadata calls.

        Parameters
        ----------
        features : str 
            Features, metadata columns, or factors (FactorN) to fetch
        """
        # If a sole variable name is used, wrap it in a list
        if not isinstance(variables, Iterable) or isinstance(variables, str):
            variables = [variables]

        # Remove None values and duplicates
        variables = [i for i in variables if i is not None]
        variables = list(set(variables))

        var_meta = list()
        var_features = list()
        var_factors = list()

        # Split all the variables into metadata and features
        for i, var in enumerate(variables):
            if var in self.metadata.columns:
                var_meta.append(var)
            elif var.capitalize().startswith("Factor"):
                # Unify factor naming
                variables[i] = var.capitalize()
                var_factors.append(var.capitalize())
            else:
                var_features.append(var)


        var_list = list()
        if len(var_meta) > 0: var_list.append(self.metadata[var_meta])
        if len(var_features) > 0: var_list.append(self.get_data(var_features, df=True))
        if len(var_factors) > 0: var_list.append(self.get_factors(factors=var_factors, df=True))

        # Return a DataFrame with columns ordered as requested
        return pd.concat(var_list, axis=1)[variables]

    def __check_views(self, views):
        return self.__check_grouping(views, "views")

    def __check_groups(self, groups):
        return self.__check_grouping(groups, "groups")

    def __check_grouping(self, groups, grouping_instance):
        assert grouping_instance in ["groups", "views"]
        # Use all groups if no specific groups are requested
        if groups is None:
            if grouping_instance == "groups":
                groups = self.groups
            elif grouping_instance == "views":
                groups = self.views
        # If a sole group name is used, wrap it in a list
        if not isinstance(groups, Iterable) or isinstance(groups, str):
            groups = [groups]
        # Do not accept boolean values
        if any([isinstance(g, bool) for g in groups]):
            if grouping_instance == "groups":
                raise ValueError(
                    f"Please provide relevant group names. Boolean values are not accepted. Group names of this model are {', '.join(self.groups)}."
                )
            elif grouping_instance == "views":
                raise ValueError(
                    f"Please provide relevant view names. Boolean values are not accepted. View names of this model are {', '.join(self.views)}."
                )
        # Convert integers to group names
        if grouping_instance == "groups":
            groups = [self.groups[g] if isinstance(g, int) else g for g in groups]
        elif grouping_instance == "views":
            groups = [self.views[g] if isinstance(g, int) else g for g in groups]
        return groups

    def __check_factors(self, factors):
        # Use all factors by default
        if factors is None:
            factors = list(range(self.nfactors))
        # If one factor is used, wrap it in a list
        if not isinstance(factors, Iterable) or isinstance(factors, str):
            factors = [factors]
        # Convert factor names (FactorN) to factor indices (N-1)
        findices = [
            int(fi.replace("Factor", "")) - 1 if isinstance(fi, str) else fi
            for fi in factors
        ]
        factors = [f"Factor{fi+1}" if isinstance(fi, int) else fi for fi in factors]

        return (findices, factors)

    def get_factor_r2(
        self, factor_index: int, 
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        views: Optional[Union[str, int, List[str], List[int]]] = None,
        group_label: Optional[str] = None, groups_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if groups_df is not None and group_label is not None:
            print("Please specify either group_label or groups_df but not both")
            sys.exit(1)

        groups = self.__check_groups(groups)
        views = self.__check_views(views)

        r2_df = pd.DataFrame()
        if groups_df is None and (group_label is None or group_label == "group"):
            for view in views:
                for group in groups:
                    crossprod = np.array(
                        self.expectations["Z"][group][[factor_index], :]
                    ).T.dot(np.array(self.expectations["W"][view][[factor_index], :]))
                    y = np.array(self.data[view][group])
                    a = np.sum((y - crossprod) ** 2)
                    b = np.sum(y ** 2)
                    r2_df = r2_df.append(
                        {
                            "View": view,
                            "Group": group,
                            "Factor": f"Factor{factor_index+1}",
                            "R2": 1 - a / b,
                        },
                        ignore_index=True,
                    )

        # When calculating for a custom set of groups,
        # Z matrix has to be merged and then split
        # according to the new grouping of samples
        else:
            custom_groups = groups_df.iloc[:, 0].unique() if group_label is None else self.samples_metadata[group_label].unique()
            if groups_df is None:
                groups_df = self.samples_metadata.loc[:,[group_label]]

            z = np.concatenate(
                [self.expectations["Z"][group][:, :] for group in groups], axis=1
            )

            z_custom = dict()
            for group in custom_groups:
                z_custom[group] = z[:, np.where(groups_df.iloc[:, 0] == group)[0]]
            del z

            for view in views:

                y_view = np.concatenate(
                    [self.data[view][group][:, :] for group in groups], axis=0
                )

                data_view = dict()
                for group in custom_groups:
                    data_view[group] = y_view[
                        np.where(groups_df.iloc[:, 0] == group)[0], :
                    ]

                for group in custom_groups:
                    crossprod = np.array(z_custom[group][[factor_index], :]).T.dot(
                        np.array(self.expectations["W"][view][[factor_index], :])
                    )
                    y = np.array(data_view[group])
                    a = np.sum((y - crossprod) ** 2)
                    b = np.sum(y ** 2)
                    r2_df = r2_df.append(
                        {
                            "View": view,
                            "Group": group,
                            "Factor": f"Factor{factor_index+1}",
                            "R2": 1 - a / b,
                        },
                        ignore_index=True,
                    )
        return r2_df

    def get_r2(
        self,
        factors: Optional[Union[int, List[int], str, List[str]]] = None,
        groups: Optional[Union[str, int, List[str], List[int]]] = None,
        views: Optional[Union[str, int, List[str], List[int]]] = None,
        groups_df: Optional[pd.DataFrame] = None,
        group_label: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get variance explained (R2) per factor, view, and group.

        factors : optional
            List of factors to consider (all by default)
        groups : optional
            List of groups to consider (all by default)
        views : optional
            List of views to consider (all by default)
        group_label : optional
            Sample (cell) metadata column to be used as group assignment
        groups_df : optional pd.DataFrame
            Data frame with samples (cells) as index and first column as group assignment
        """
        if "variance_explained" in self.model.keys() and group_label is None and groups_df is None:
            # Load from file if pre-computed
            r2 = pd.concat([
             pd.DataFrame(r2, index=self.views, columns=[f"Factor{i+1}" for i in range(self.nfactors)])\
                .rename_axis("View").reset_index().melt(id_vars=["View"], var_name="Factor", value_name="R2")\
                .assign(Group=group)\
                .loc[:,["Factor", "View", "Group", "R2"]]
                  for group, r2 in self.model["variance_explained"]["r2_per_factor"].items()])
            if factors is not None:
                _, factors = self.__check_factors(factors)
                r2 = r2[r2.Factor.isin(factors)]
            if groups is not None:
                groups = self.__check_groups(groups)
                r2 = r2[r2.Group.isin(groups)]
            if views is not None:
                view = self.__check_views(views)
                r2 = r2[r2.View.isin(views)]
        else:        
            if groups_df is not None and group_label is not None:
                print("Please specify either group_label or groups_df but not both")
                sys.exit(1)
            r2 = pd.DataFrame()
            findices, _ = self.__check_factors(factors)
            for fi in findices:
                r2 = r2.append(self.get_factor_r2(fi, groups=groups, views=views, group_label=group_label, groups_df=groups_df))
        return r2

    def get_factor_r2_null(
        self,
        factor_index: int,
        groups_df: Optional[pd.DataFrame],
        group_label: Optional[str],
        n_iter=100,
        return_full=False,
        return_true=False,
        return_pvalues=True,
        fdr=True,
    ) -> pd.DataFrame:
        r2_df = pd.DataFrame()

        if groups_df is None and group_label is None:
            group_label = "group"

        if groups_df is None:
            groups_df = self.samples_metadata.loc[:,[group_label]]

        custom_groups = groups_df.iloc[:, 0].unique()

        z = np.concatenate(
            [self.expectations["Z"][group][:, :] for group in self.groups], axis=1
        )

        for i in range(n_iter + 1):
            # Canculate true group assignment for iteration 0
            if i > 0:
                groups_df.iloc[:, 0] = groups_df.iloc[:, 0].sample(frac=1).values

            z_custom = dict()
            for group in custom_groups:
                z_custom[group] = z[:, np.where(groups_df.iloc[:, 0] == group)[0]]

            for view in self.views:

                y_view = np.concatenate(
                    [self.data[view][group][:, :] for group in self.groups], axis=0
                )

                data_view = dict()
                for group in custom_groups:
                    data_view[group] = y_view[
                        np.where(groups_df.iloc[:, 0] == group)[0], :
                    ]

                for group in custom_groups:
                    crossprod = np.array(z_custom[group][[factor_index], :]).T.dot(
                        np.array(self.expectations["W"][view][[factor_index], :])
                    )
                    y = np.array(data_view[group])
                    a = np.sum((y - crossprod) ** 2)
                    b = np.sum(y ** 2)
                    r2_df = r2_df.append(
                        {
                            "View": view,
                            "Group": group,
                            "Factor": f"Factor{factor_index+1}",
                            "R2": 1 - a / b,
                            "Iteration": i,
                        },
                        ignore_index=True,
                    )

        if return_full:
            if return_true:
                return r2_df
            else:
                return r2_df[r2_df.Iteration != 0].reset_index(drop=True)

        r2_obs = r2_df[r2_df.Iteration == 0]
        r2_df = r2_df[r2_df.Iteration != 0]

        if not return_pvalues:
            r2_null = r2_df.groupby(["Factor", "Group", "View"]).agg(
                {"R2": ["mean", "std"]}
            )
            return r2_null.reset_index()

        r2_pvalues = pd.DataFrame(
            r2_obs.set_index(["Group", "View", "Factor"])
            .loc[:, ["R2"]]
            .join(r2_df.set_index(["Group", "View", "Factor"]), rsuffix="_null")
            .groupby(["Group", "View", "Factor"])
            .apply(lambda x: np.mean(x["R2"] <= x["R2_null"]))
        )
        r2_pvalues.columns = ["PValue"]

        if fdr:
            r2_pvalues["FDR"] = padjust_fdr(r2_pvalues.PValue)
            return r2_pvalues.reset_index().sort_values("FDR", ascending=True)
        else:
            return r2_pvalues.reset_index().sort_values("PValue", ascending=True)

    def get_r2_null(
        self,
        factors: Union[int, List[int], str, List[str]] = None,
        n_iter: int = 100,
        groups_df: Optional[pd.DataFrame] = None,
        group_label: Optional[str] = None,
        return_full=False,
        return_pvalues=True,
        fdr=True,
    ) -> pd.DataFrame:
        findices, factors = self.__check_factors(factors)
        r2 = pd.DataFrame()
        for fi in findices:
            r2 = r2.append(
                self.get_factor_r2_null(
                    fi,
                    groups_df=groups_df,
                    group_label=group_label,
                    n_iter=n_iter,
                    return_full=return_full,
                    return_pvalues=return_pvalues,
                    fdr=fdr,
                )
            )
        return r2

    def project_data(
        self,
        data,
        view: Union[str, int] = None,
        factors: Union[int, List[int], str, List[str]] = None,
        df: bool = False,
        feature_intersection: bool = False,
    ):
        """
        Project new data onto the factor space of the model.

        For the projection, a pseudo-inverse of the weights matrix is calculated 
        and its product with the provided data matrix is calculated.

        Parameters
        ----------
        data
            Numpy array or Pandas DataFrame with the data matching the number of features
        view : optional
            A view of the model to consider (first view by default)
        factors : optional
            Indices of factors to use for the projection (all factors by default)
        """
        if view is None:
            view = 0
        view = self.__check_views([view])[0]
        findices, factors = self.__check_factors(factors)

        # Calculate the inverse of W
        winv = np.linalg.pinv(self.get_weights(views=view, factors=factors))

        # Find feature intersection to match the dimensions
        if feature_intersection:
            if data.shape[1] != self.shape[1] and isinstance(data, pd.DataFrame):
                fs_common = np.intersect1d(data.columns.values, self.features[view])
                data = data.loc[:, fs_common]

                # Get indices of the common features in the original data
                f_sorted = np.argsort(self.features[view])
                fs_common_pos = np.searchsorted(
                    self.features[view][f_sorted], fs_common
                )
                f_indices = f_sorted[fs_common_pos]

                winv = winv[:, f_indices]
                warnings.warn(
                    "Only {} features are matching between two datasets of size {} (original data) and {} (projected data).".format(
                        fs_common.shape[0], self.shape[1], data.shape[1]
                    )
                )

        # Predict Z for the provided data
        zpred = np.dot(data, winv.T)

        if df:
            zpred = pd.DataFrame(zpred)
            zpred.columns = factors
            if isinstance(data, pd.DataFrame):
                zpred.index = data.index
        return zpred


# Utility functions


def umap(data, n_neighbors=10, spread=1):
    """
    Run UMAP on a provided matrix or data frame

    Parameters
    ----------
    data
        Numpy array or Pandas DataFrame with data to run UMAP on (samples in rows)
    n_neighbors : optional
        n_neighbors parameter for UMAP
    spread : optional
        spread parameter for UMAP
    """
    import umap

    embedding = umap.UMAP(n_neighbors=n_neighbors, spread=spread).fit_transform(data)

    if isinstance(data, pd.DataFrame):
        embedding = pd.DataFrame(embedding)
        embedding.columns = ["UMAP1", "UMAP2"]
        embedding.index = data.index

    return embedding


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

    ranked_p_values = rankdata(mx).reshape((-1,mx.shape[1]))
    fdr = mx * mx.shape[0] * mx.shape[1] / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr
