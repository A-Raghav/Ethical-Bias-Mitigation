"""This module contains the wrapper class around the aif360 (AI Fairness 360) 
module. This class can be used to -
    1. detect unprivileged groups suffering from ethical bias
    2. visualise the bias in mitigated vs unmitigated models (#TODO: WIP)
    3. mitigate the bias (#TODO: WIP)

The user needs to -
    1. provide the raw pandas dataset as input
    2. provide the favourable value for label (y-variable)
    3. select the sensitive features for which they need to perform the bias 
       detection (and subsequent mitgation (#TODO: WIP))
    4. select the mitigation_strategy. This can be - 
        4a. Preprocessing - "reweighing"
        4b. Postprocessing - "reject_option_classification"
        (#TODO: More mitigation strategies to be added later) 

NOTE that the dataset needs to be - 
    * structured dataset
    * binary labelled (binary target variable)
    * in raw form (this class performs the OHE itself, so no need to do that beforehand)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)


class BiasAndFairnessMitigation:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        positive_outcome: str,
        mitigation_strategy: str,
    ):
        if positive_outcome not in y.unique():
            raise ValueError(
                f'"{positive_outcome}" value does not exist in "target" column.'
            )
        self.positive_outcome = positive_outcome
        self.mitigation_strategy = mitigation_strategy
        self.X_raw = X
        self.y_raw = y
        self.features = X.columns
        self.X = self._configure_X(X)
        self.y = self._configure_y(y)
        self.protected_attributes: list = []
        self.sensitive_features: dict = {}
        self.bias_scores: dict = {}
        self.y_pred_unmit = None
        self.y_pred_mit = None

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

    def _create_standard_dataset(
        self,
        df: pd.DataFrame,
        protected_attribute_name: str,
        label_name="class",
    ) -> StandardDataset:
        return StandardDataset(
            df,
            label_name=label_name,
            favorable_classes=[1],
            protected_attribute_names=[protected_attribute_name],
            privileged_classes=[[1]],
        )

    def _apply_reweighing(self, dataset, privileged_groups, unprivileged_groups):
        # Step 1: Split dataset into train:test :: 7:3
        dataset_train, dataset_test = dataset.split([0.7], shuffle=True, seed=42)

        # Step 2: Initialise scaler and LogReg models
        scaler = MinMaxScaler()
        model = LogisticRegression(solver="liblinear", random_state=42)

        # Step 3: Compute sample (instance) weights
        RW = Reweighing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )
        dataset_train_rw = RW.fit_transform(dataset_train)
        weights = dataset_train_rw.instance_weights

        # Step 4: Fit model without sample weights & compute metrics
        X_train = scaler.fit_transform(dataset_train.features)
        y_train = dataset_train.labels.ravel()
        X_test = scaler.transform(dataset_test.features)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        dataset_test_pred = dataset_test.copy()
        dataset_test_pred.labels = y_test_pred
        metric_unmit = ClassificationMetric(
            dataset_test,
            dataset_test_pred,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )

        # Step 5: Fit model with sample weights & compute metrics
        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )
        y_test_pred_transf = model.predict(X_test)
        dataset_test_pred_transf = dataset_test.copy()
        dataset_test_pred_transf.labels = y_test_pred_transf
        metric_mit = ClassificationMetric(
            dataset_test,
            dataset_test_pred_transf,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )

        return metric_unmit, metric_mit

    def _apply_reject_option(self, dataset, privileged_groups, unprivileged_groups):
        dataset_train, dataset_vt = dataset.split([0.7], shuffle=True, seed=42)
        dataset_valid, dataset_test = dataset_vt.split([0.5], shuffle=True, seed=42)

        SCALER = StandardScaler()
        X_train = SCALER.fit_transform(dataset_train.features)
        y_train = dataset_train.labels.ravel()

        LR = LogisticRegression()
        LR.fit(X_train, y_train)

        thresh = 0.5
        pos_ind = np.where(LR.classes_ == dataset_train.favorable_label)[0][0]

        dataset_valid_pred = dataset_valid.copy(deepcopy=True)
        X_valid = SCALER.fit_transform(dataset_valid_pred.features)
        dataset_valid_pred.scores = LR.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)
        dataset_valid_pred.labels = dataset_valid_pred.scores > thresh

        dataset_test_pred = dataset_test.copy(deepcopy=True)
        X_test = SCALER.fit_transform(dataset_test_pred.features)
        dataset_test_pred.scores = LR.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
        dataset_test_pred.labels = dataset_test_pred.scores > thresh

        ROC = RejectOptionClassification(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            low_class_thresh=0.01,
            high_class_thresh=0.99,
            num_class_thresh=100,
            num_ROC_margin=50,
            metric_name="Statistical parity difference",
            metric_ub=0.05,
            metric_lb=-0.05,
        )
        ROC = ROC.fit(dataset_valid, dataset_valid_pred)
        metric_unmit = ClassificationMetric(
            dataset_test,
            dataset_test_pred,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )
        # Metrics for the transformed test set
        dataset_test_pred_transf = ROC.predict(dataset_test_pred)

        metric_mit = ClassificationMetric(
            dataset_test,
            dataset_test_pred_transf,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )

        return metric_unmit, metric_mit

    def _compute_scores(self) -> dict:
        """Computes the bias-related scores relative to the sensitive features.

        The following scores are computed -
            1. Disparate-Impact score

        Returns:
            dict: Dict of Disparate-Impact scores
        """
        for col, val in self.sensitive_features.items():
            protected_attribute_name = f"{col}_{val}"

            # Step 1: Create the StandardDataset object
            dataset = self._create_standard_dataset(
                pd.concat([self.X, self.y], axis=1), protected_attribute_name
            )

            # Step 2: Define privileged and unprivileged groups
            privileged_groups = [{protected_attribute_name: 1.0}]
            unprivileged_groups = [{protected_attribute_name: 0.0}]

            # Step 3: Apply a mitigation strategy to create optimised train and test datasets
            if self.mitigation_strategy == "reweighing":
                metric_unmit, metric_mit = self._apply_reweighing(
                    dataset, privileged_groups, unprivileged_groups
                )

            elif self.mitigation_strategy == "reject_option_classification":
                metric_unmit, metric_mit = self._apply_reject_option(
                    dataset, privileged_groups, unprivileged_groups
                )

            # Step 5: Compute bias scores
            self.bias_scores[protected_attribute_name] = {
                "unmitigated": {
                    "di_score": metric_unmit.disparate_impact(),
                    "spd_score": metric_unmit.statistical_parity_difference(),
                    "accuracy": metric_unmit.accuracy(),
                },
                "mitigated": {
                    "di_score": metric_mit.disparate_impact(),
                    "spd_score": metric_mit.statistical_parity_difference(),
                    "accuracy": metric_mit.accuracy(),
                },
            }

        return self.bias_scores

    def describe_bias_and_fairness(self, sensitive_features: dict):
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

    def plot_bias_scores(self, sensitive_feature):
        privileged_group = self.sensitive_features[sensitive_feature]
        protected_attribute_name = f"{sensitive_feature}_{privileged_group}"

        # prepare data for bar-plot
        df = pd.DataFrame(self.bias_scores[protected_attribute_name])
        di_data = df.T.to_dict()["di_score"]
        spd_data = df.T.to_dict()["spd_score"]
        acc_data = df.T.to_dict()["accuracy"]

        # create figure
        plt.style.use("fivethirtyeight")
        fig = plt.figure(figsize=(10, 5))

        # add subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(
            di_data.keys(),
            di_data.values(),
            width=0.7,
            color=["lightcoral", "lightsteelblue"],
        )
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(
            spd_data.keys(),
            spd_data.values(),
            width=0.7,
            color=["lightcoral", "lightsteelblue"],
        )

        # set xlim and ylim
        ax1.set_xlim(-0.6, 1.5)
        ax1.set_ylim(0, 2)
        ax2.set_xlim(-0.6, 1.5)
        ax2.set_ylim(-1, 1)

        # set tick-labels
        ax1.set_xticklabels(["", ""])
        ax1.set_yticklabels(labels=[0, "", 0.5, "", 1, "", 1.5, "", 2])
        ax2.set_xticklabels(["", ""])
        ax2.set_yticklabels(labels=[-1, "", -0.5, "", 0, "", 0.5, "", 1])

        # set axhlines and axvlines
        ax1.axhline(0.016, color="black", lw=1)
        ax1.axhline(1, color="black", lw=1, ls="-.", alpha=0.5)
        ax1.axvline(-0.5, color="black", lw=1)
        ax2.axhline(0, color="black", lw=1, ls="-.", alpha=0.5)
        ax2.axhline(-0.984, color="black", lw=1)
        ax2.axvline(-0.5, color="black", lw=1)

        # set legend
        legend_elements = [
            Patch(facecolor="lightcoral", label="Unmitigated"),
            Patch(facecolor="lightsteelblue", label="Mitigated"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(0.9, 1.25))

        # set patch
        poly_coords_1 = [(-0.5, 1.25), (1.5, 1.25), (1.5, 0.8), (-0.5, 0.8)]
        poly_coords_2 = [(-0.5, 0.1), (1.5, 0.1), (1.5, -0.1), (-0.5, -0.1)]
        ax1.add_patch(plt.Polygon(poly_coords_1, color="black", alpha=0.1))
        ax2.add_patch(plt.Polygon(poly_coords_2, color="black", alpha=0.1))

        # set text
        ax1.text(s=round(di_data["unmitigated"], 2), x=-0.1, y=0.05)
        ax1.text(s=round(di_data["mitigated"], 2), x=0.9, y=0.05)
        ax2.text(s=round(spd_data["unmitigated"], 2), x=-0.1, y=0.05)
        ax2.text(s=round(spd_data["mitigated"], 2), x=0.9, y=0.05)

        acc_unmit = round(acc_data["unmitigated"], 3)
        acc_mit = round(acc_data["mitigated"], 3)
        ax1.text(
            s=f"Protected Attribute: {sensitive_feature}",
            x=-0.5,
            y=2.6,
            verticalalignment="top",
            size=20,
            weight="bold",
        )
        ax1.text(
            s=f"Privileged Group: {privileged_group}",
            x=-0.5,
            y=2.4,
            verticalalignment="top",
        )
        ax1.text(
            s=f"Accuracy after mitigation changed from {acc_unmit*100}% to {acc_mit*100}%",
            x=-0.5,
            y=2.3,
            verticalalignment="top",
        )
        ax1.text(
            s=f"Bias mitigation algorithm applied: {self.mitigation_strategy}",
            x=-0.5,
            y=2.2,
            verticalalignment="top",
        )

        ax1.text(
            s="Disparate Impact",
            x=0,
            y=-0.1,
            verticalalignment="top",
            horizontalalignment="left",
            weight="bold",
        )
        ax2.text(
            s="Statistical Parity Difference",
            x=-0.3,
            y=-1.1,
            verticalalignment="top",
            horizontalalignment="left",
            weight="bold",
        )
        s = """
        Computed as the ratio of rate of favorable outcome for 
        the unprivileged group to that of the privileged group.

        The ideal value of this metric (DI) is 1.0. 

        DI < 1 implies a higher benefit for the privileged group.
        DI > 1 implies a higher benefit for the unprivileged group.

        Fairness for this metric is between 0.8 and 1.25
        """
        ax1.text(
            s=s,
            x=-0.7,
            y=-0.2,
            verticalalignment="top",
            horizontalalignment="left",
            size=10,
            linespacing=1.5,
        )
        s = """
        Computed as the difference of the rate of favorable 
        outcomes received by the unprivileged group to the 
        privileged group.

        The ideal value of this metric is 0.

        Fairness for this metric is between -0.1 and 0.1
        """
        ax2.text(
            s=s,
            x=-0.7,
            y=-1.2,
            verticalalignment="top",
            horizontalalignment="left",
            size=10,
            linespacing=1.5,
        )
        ax1.annotate(
            "",
            xy=(0, -0.12),
            xycoords="axes fraction",
            xytext=(1.05, -0.12),
            arrowprops=dict(arrowstyle="-", color="gray"),
            alpha=0.3,
        )
        ax1.annotate(
            "",
            xy=(1.175, -0.12),
            xycoords="axes fraction",
            xytext=(2.225, -0.12),
            arrowprops=dict(arrowstyle="-", color="gray"),
            alpha=0.3,
        )

        plt.show()
