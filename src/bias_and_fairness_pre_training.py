"""This module contains the wrapper class around the aif360 (AI Fairness 360) 
module. This class can be used to -
    1. detect unprivileged groups suffering from ethical bias
    2. mitigate the bias (#TODO: WIP)
    3. visualise the bias in mitigated vs unmitigated models (#TODO: WIP)

The user needs to -
    1. provide the pandas dataset as input
    2. select the privileged groups on which they need to perform the bias 
       detection and subsequent mitigation

NOTE that the dataset needs to be - 
    * structured dataset
    * binary labelled (binary target variable)
"""
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.explainers import MetricTextExplainer
from aif360.metrics import BinaryLabelDatasetMetric


class BiasAndFairnessPreTraining:
    def __init__(self, X: pd.DataFrame, y: pd.Series, positive_outcome: str):
        if positive_outcome not in y.unique():
            raise ValueError(
                f'"{positive_outcome}" value does not exist in "target" column.'
            )
        self.positive_outcome = positive_outcome
        self.X_raw = X
        self.y_raw = y
        self.features = X.columns
        self.X = self._configure_X(X)
        self.y = self._configure_y(y)
        self.protected_attributes: list = []
        self.sensitive_features: dict = {}
        self.bias_scores: dict = {}

    def _configure_X(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Converts the categorical columns of X into their dummies (OHE)

        Args:
            X_raw (pd.DataFrame): Raw X

        Returns:
            pd.DataFrame: Processed X
        """
        return pd.get_dummies(X_raw)

    def _configure_y(self, y_raw: pd.Series) -> pd.Series:
        """Converts the categorical binary target column into 1/0 binary.
        The values corresponding to the positive outcome become 1, and
        vice versa.

        Args:
            X_raw (pd.DataFrame): Raw X

        Returns:
            pd.DataFrame: Processed X
        """
        if y_raw.nunique() > 2:
            raise ValueError(
                """The dataset has more than 2 labels. The process is configured only for a Binary label dataset for now."""
            )
        return (y_raw == self.positive_outcome) * 1

    def return_categorical_column_names(self) -> list:
        """Returns the names of categorical columns to user.
        This is a utility function

        Returns:
            list: list of names of categorical columns
        """
        cat_columns = []

        for _, column in self.X_raw.items():
            if column.dtype == "category":
                cat_columns.append(column.name)

        return cat_columns

    def _protected_attributes(self, protected_attributes: list) -> list:
        """Checks if the protected attributes exist in the features (columns of X)

        Args:
            protected_attributes (list): List of names of protected attributes

        Raises:
            ValueError: Atleast one protexted attribute is not present in X

        Returns:
            list: List of validated protected attributes
        """
        missing_columns = [
            attr for attr in protected_attributes if attr not in self.features
        ]

        if missing_columns:
            raise ValueError(
                f"Atleast one protected attribute - (\{missing_columns}\) not present in the input dataframe."
            )

        return protected_attributes

    def _compute_scores(self) -> dict:
        """Computes the bias-related scores relative to the sensitive features.

        The following scores are computed -
            1. Disparity-Impact score

        Returns:
            dict: Dict of Disparity-Impact scores
        """
        label_name = "class"
        df = self.X.copy()
        df[label_name] = self.y

        # Loop through sensitive-features
        for col, val in self.sensitive_features.items():
            protected_attribute_name = f"{col}_{val}"

            # Step 1: Create the StandardDataset object
            dataset = StandardDataset(
                df,
                label_name=label_name,
                favorable_classes=[1],
                protected_attribute_names=[protected_attribute_name],
                privileged_classes=[[1]],
            )

            # Step 2: Define privileged and unprivileged groups
            privileged_groups = [{protected_attribute_name: 1.0}]
            unprivileged_groups = [{protected_attribute_name: 0.0}]

            # Step 3: Create BinaryLabelDatasetMetric object
            metric = BinaryLabelDatasetMetric(
                dataset,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )

            # Step 4: Compute disparity-impact score and statistical parity
            #         difference for the given sensitive feature
            di_score = metric.disparate_impact()
            spd_score = metric.statistical_parity_difference()
            self.bias_scores[protected_attribute_name] = {
                "di_score": di_score,
                "spd_score": spd_score,
            }

            # Step 5: Create Explainer for explaining the metrics
            explainer = MetricTextExplainer(metric)
            print(
                f"Explaining bias-scores for {val} privileged group of {col} protected attribute -"
            )
            print(explainer.disparate_impact())
            print(explainer.statistical_parity_difference(), "\n")

        return self.bias_scores

    def specify_sensitive_features(self, sensitive_features: dict):
        """Checks if the protected attributes exist in the features (columns of X)

        Args:
            sensitive_features (dict): Dict of names of sensitive features.
            The dictionary contains key:value pairs of protected attributes and
            privilege groups.

            For example, if `race` is the protected attribute, and `White` is the
            privilege group, the the sensitive_features dictionary will be -

                            {
                                'race': 'White'
                            }
        """
        self.protected_attributes = self._protected_attributes(
            sensitive_features.keys()
        )
        self.sensitive_features = sensitive_features
        return self._compute_scores()
