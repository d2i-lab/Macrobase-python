import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


class Macrobase:
    """A Macrobase implementation.
    Typical usage example:
        macrobase=Macrobase(df, min_support=0.1, min_rr=0.1)
        result=macrobase.fit()
    """

    def __init__(self,
                 df, min_support,
                 min_rr, usecolnames=True,
                 max_len=None, invert=False):
        """Initializes the outliers, dataset, support, and risk ratio.
        Args:
            df (pd.DataFrame):
                A pandas dataframe of the complete dataset with
                an additional 'Outlier' column of booleans indicating
                whether a row is an outlier or not.
                The dataframe should not contain any Label or Predicted columns.
            min_support (double): A double of minimum support threshold.
            min_rr (double): A double of minimum risk ratio threshold.
            usecolnames (bool, optional):
                A boolean of whether to use column names. Defaults to True.
            max_len (int, optional):
                An int of the maximum length of the itemsets. Defaults to None.
            invert (boolean):
                If True, the accuracy is calculated as (outliers/all). 
                Otherwise, default to (1-outliers/all).
        """

        self.outlier = []
        self.df = df.copy()
        self.min_support = min_support
        self.min_rr = min_rr
        self.use_colnames = usecolnames
        self.max_len = max_len
        self.invert = invert

    def data_preprocess(self):
        """Preprocesses the pandas dataframe `self.df`
        by converting each column to the format 'column_name=value'.
        """
        df = [[col + "=" + str(val) for col, val in zip(self.df.columns, row)]
              for row in self.df.values]
        te = TransactionEncoder()
        te_ary = te.fit(df).transform(df)
        df_transformed = pd.DataFrame(te_ary, columns=te.columns_)
        outliers = df_transformed.loc[(df_transformed['Outlier=True'])]

        self.df = df_transformed.drop(
            ['Outlier=True', 'Outlier=False'], axis=1)
        self.outlier = outliers.drop(['Outlier=True', 'Outlier=False'], axis=1)

    def generate_new_combinations(self, old_combinations):
        """
        Generator of all combinations based on the last state of Apriori algorithm
        Parameters
        -----------
        old_combinations: np.array
            All combinations with enough support in the last step
            Combinations are represented by a matrix.
            Number of columns is equal to the combination size
            of the previous step.
            Each row represents one combination
            and contains item type ids in the ascending order
            ```
                    0        1
            0      15       20
            1      15       22
            2      17       19
            ```
        Returns
        -----------
        Generator of all combinations from the last step x items
        from the previous step.
        """
        items_types_in_previous_step = np.unique(old_combinations.flatten())
        for old_combination in old_combinations:
            max_combination = old_combination[-1]
            mask = items_types_in_previous_step > max_combination
            valid_items = items_types_in_previous_step[mask]
            old_tuple = tuple(old_combination)
            for item in valid_items:
                yield from old_tuple
                yield item

    def fit(self):
        """Find frequent itemsets with support >= minimum support
          and risk ratio >= minimum risk ratio threshold.
        Returns:
            pd.DataFrame: frequent itemsets with support >= minimum support
              and risk ratio >= minimum risk ratio threshold.
        """
        self.data_preprocess()
        df_all = self.df.values
        df_outlier = self.outlier.values
        df_rows_count = float(df_all.shape[0])
        outlier_rows_count = float(df_outlier.shape[0])

        def _support(_x, _n_rows):
            """DRY private method to calculate support as
            the row-wise sum of values / number of rows
             Args:
                 _x : matrix of bools or binary
                 _n_rows : numeric, number of rows in _x
                 _is_sparse : bool True if _x is sparse
             Returns:
                 np.array, shape = (n_rows, )
             """

            out = np.sum(_x, axis=0) / _n_rows
            return np.array(out).reshape(-1)

        def _risk_ratio(_outliers, _outlier_rows, _all, _all_rows):
            """Calculate the risk ratio as (ao/(ao+ai))/(bo/(bo+bi)), where:
            - ao is the number of times the given attribute combination
              appears in outliers
            - ai is the number of times the given attribute combination
              appears in inliers
            - bo is the number of other outliers
            - bi is the number of other inliers
            Args:
                _outliers (pd.DataFrame): A pandas dataframe of outliers.
                _outlier_rows (int): An int of number of rows in _outliers.
                _all (pd.DataFrame): A pandas dataframe of the dataset.
                _all_rows (int): An int of number of rows in _all.
            Returns:
                np.array, shape = (n_rows, )
            """
            ao = np.sum(_outliers, axis=0)
            aoi = np.sum(_all, axis=0)
            bo = _outlier_rows-ao
            boi = _all_rows-aoi
            return np.array((ao/aoi)/(bo/boi)).reshape(-1)

        def _accuracy(_outliers, _all):
            """Calculate the accuracy as (outliers/all) or 1-(outliers/all)
            Args:
                _outliers (pd.DataFrame): A pandas dataframe of outliers.
                _all (pd.DataFrame): A pandas dataframe of the dataset.
            Returns:
                np.array, shape = (n_rows, )
            """
            ao = np.sum(_outliers, axis=0)
            all = np.sum(_all, axis=0)

            if self.invert:
                return np.array(ao/all).reshape(-1)
            else:
                return np.array(1-ao/all).reshape(-1)

        # calculating support and risk ratio
        acc = _accuracy(df_outlier, df_all)
        support = _support(df_outlier, outlier_rows_count)
        risk_ratio = _risk_ratio(
            df_outlier, outlier_rows_count, df_all, df_rows_count)

        # creating mask
        # [T,T, F, T,...]
        masksupport1 = (support >= self.min_support).reshape(-1)
        maskrr1 = (risk_ratio >= self.min_rr).reshape(-1)
        mask1 = np.logical_and(masksupport1, maskrr1)

        # assign index to each itemset
        ary_col_idx = np.arange(df_outlier.shape[1])

        # creating dictionary for frequent itemset and their corresponding support and risk ratio
        support_dict = {1: support[mask1]}
        acc_dict = {1: acc[mask1]}
        risk_ratio_dict = {1: risk_ratio[mask1]}
        itemset_dict = {1: ary_col_idx[masksupport1].reshape(-1, 1)}
        result_dict = {1: ary_col_idx[mask1].reshape(-1, 1)}

        max_itemset = 1

        while max_itemset and max_itemset < (self.max_len or float("inf")):
            next_max_itemset = max_itemset + 1
            combin = self.generate_new_combinations(itemset_dict[max_itemset])
            combin = np.fromiter(combin, dtype=int)
            # size=next_max_itemset itemsets
            combin = combin.reshape(-1, next_max_itemset)

            if combin.size == 0:
                break

            _bools_outlier = np.all(df_outlier[:, combin], axis=2)
            _bools_all = np.all(df_all[:, combin], axis=2)
            bools_outlier_rows_count = float(_bools_outlier.shape[0])
            bools_all_rows_count = float(_bools_all.shape[0])

            acc = _accuracy(np.array(_bools_outlier), np.array(_bools_all))
            support = _support(np.array(_bools_outlier),
                               bools_outlier_rows_count)
            risk_ratio = _risk_ratio(np.array(_bools_outlier),
                                     bools_outlier_rows_count,
                                     np.array(_bools_all),
                                     bools_all_rows_count)
            # creating mask
            # [T,T, F, T,...]
            masksupport = (support >= self.min_support).reshape(-1)
            maskrr = (risk_ratio >= self.min_rr).reshape(-1)
            mask = np.logical_and(masksupport, maskrr)

            itemset_dict[next_max_itemset] = np.array(combin[masksupport])
            result_dict[next_max_itemset] = np.array(combin[mask])
            support_dict[next_max_itemset] = np.array(support[mask])
            risk_ratio_dict[next_max_itemset] = np.array(risk_ratio[mask])
            acc_dict[next_max_itemset] = np.array(acc[mask])
            max_itemset = next_max_itemset

        all_res = []
        for k in sorted(result_dict):
            support = pd.Series(support_dict[k])
            risk_ratio = pd.Series(risk_ratio_dict[k])
            acc = pd.Series(acc_dict[k])
            itemsets = pd.Series([frozenset(i)
                                 for i in result_dict[k]], dtype="object")

            res = pd.concat((support, acc, risk_ratio, itemsets), axis=1)
            all_res.append(res)

        res_df = pd.concat(all_res)
        res_df.columns = ["support", "accuracy", "risk ratio", "itemsets"]
        if self.use_colnames:
            mapping = {idx: item for idx, item in enumerate(self.df.columns)}
            res_df["itemsets"] = res_df["itemsets"].apply(
                lambda x: frozenset([mapping[i] for i in x])
            )
        res_df = res_df.reset_index(drop=True)

        return res_df
