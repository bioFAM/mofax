import h5py
import numpy as np
import pandas as pd

from typing import Union, List
from collections.abc import Iterable

class mofa_model():
    """Class around HDF5-based model on disk.

    This class is a thin wrapper for the HDF5 file where the trained MOFA+ model is stored.
    It also provides utility functions to get factors, weights, features, and cells info
    in the form of Pandas dataframes, and data as a NumPy array.
    """
    def __init__(self, filepath, mode="r"):
        self.filepath = filepath
        self.model = h5py.File(filepath, mode)
        
        self.data = self.model["data"]
        
        self.cells = {g:np.array(self.model["samples"][g]).astype("str") for g in self.model["samples"]}
        self.features = {m:np.array(self.model["features"][m]).astype("str") for m in self.model["features"]}
        
        self.groups = list(self.model["samples"].keys())
        self.views = list(self.model["features"].keys())
        
        self.expectations = self.model["expectations"]
        self.factors = self.model["expectations"]["Z"]
        self.weights = self.model["expectations"]["W"]
        
        self.shape = (sum(self.data[self.views[0]][group].shape[0] for group in self.groups),
                      sum(self.data[view][self.groups[0]].shape[1] for view in self.views))
        self.nfactors = self.model["expectations"]["Z"][self.groups[0]].shape[0]
        
        self.likelihoods = np.array(self.model["model_options"]["likelihoods"]).astype("str").tolist()
        
        # TODO: Update according to the latest API
        self.training_opts = {"maxiter": self.model["training_opts"][0]}

    def close(self):
        """Close the connection to the HDF5 file"""
        if self.model.__bool__():  # if the connection is still open
            self.model.close()
    
    def get_shape(self, groups=None, views=None):
        """
        Get the shape of all the data, cells and features pulled across groups and views.

        Parameters
        ----------
        groups : optional
            List of groups to consider
        views : optional
            List of views to consider
        """
        groups = self.__check_groups(groups)
        views = self.__check_views(views)
        shape = (sum(self.data[self.views[0]][group].shape[0] for group in groups),
                 sum(self.data[view][self.groups[0]].shape[1] for view in views))
        return shape
        
    def get_cells(self, groups=None):
        """
        Get the cell metadata table (cell ID and its respective group)

        Parameters
        ----------
        groups : optional
            List of groups to consider
        """
        groups = self.__check_groups(groups)
        return pd.DataFrame([[group, cell] for group, cell_list in self.cells.items() for cell in cell_list if group in groups], columns=["group", "cell"])

    def get_features(self, views=None):
        """
        Get the features metadata table (feature name and its respective view)

        Parameters
        ----------
        views : optional
            List of views to consider
        """
        views = self.__check_views(views)
        return pd.DataFrame([[view, feature] for view, feature_list in self.features.items() for feature in feature_list if view in views], columns=["view", "feature"])
    
    def get_factors(self, groups: Union[str, int, List[str], List[int]] = None,
                    factors: Union[int, List[int]] = None, df = False):
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
        """
        groups = self.__check_groups(groups)
        findices, factors = self.__check_factors(factors)
        z = np.concatenate(tuple(np.array(self.factors[group]).T[:,findices] for group in groups))
        if df:
            z = pd.DataFrame(z)
            z.columns = factors
            z.index = np.concatenate(tuple(self.cells[g] for g in groups))
        return z
    
    def get_weights(self, views: Union[str, int, List[str], List[int]] = None,
                    factors: Union[int, List[int]] = None, df = False):
        """
        Get the matrix with loadings as a NumPy array or as a DataFrame (df=True).

        Parameters
        ----------
        views : optional
            List of views to consider
        factors : optional
            Indices of factors to use
        df : optional
            Boolean value if to return W matrix as a DataFrame
        """
        views = self.__check_views(views)
        findices, factors = self.__check_factors(factors)
        w = np.concatenate(tuple(np.array(self.weights[view]).T[:,findices] for view in views))
        if df:
            w = pd.DataFrame(w)
            w.columns = factors
            w.index = np.concatenate(tuple(self.features[m] for m in views))
        return w
    
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
                raise ValueError(f"Please provide relevant group names. Boolean values are not accepted. This model's group names are {', '.join(self.groups)}.")
            elif grouping_instance == "views":
                raise ValueError(f"Please provide relevant view names. Boolean values are not accepted. This model's view names are {', '.join(self.views)}.")
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
        findices = [int(fi.replace("Factor", ""))-1 if isinstance(fi, str) else fi for fi in factors]  
        factors = [f"Factor{fi+1}" if isinstance(fi, int) else fi for fi in factors]  

        return (findices, factors)




def get_factor_r2(model: mofa_model, factor_index: int) -> pd.DataFrame:
    r2_df = pd.DataFrame()
    for view in model.views:
        for group in model.groups:
            crossprod = np.array(model.expectations["Z"][group][[factor_index],:]).T.dot(np.array(model.expectations["W"][view][[factor_index],:]))
            y = np.array(model.data[view][group])
            a = np.sum((y - crossprod)**2)
            b = np.sum(y ** 2)
            r2_df = r2_df.append({"View": view,
                          "Group": group,
                          "Factor": f"Factor{factor_index+1}",
                          "R2": 1 - a/b},
                         ignore_index=True)
    return r2_df

def get_r2(model: mofa_model, factors: Union[int, List[int], str, List[str]] = None) -> pd.DataFrame:
    findices, factors = self.__check_factors(factors)
    r2 = pd.DataFrame()
    for fi in factors:
        r2 = r2.append(get_factor_r2(model, fi))
    return r2
