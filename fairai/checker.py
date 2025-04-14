import pandas as pd
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairai.models.preprocessing import reweighing
from fairai.models.postprocessing import (
    reject_option_classification,
    calibrated_eq_odds,
    equality_of_odds,
)
from fairai.utils.metrics import bias_fairness_report


class FairAIChecker:
    """
    This class is a wrapper class around AIF-360 library used for bias and fairness.
    Contains methods that make it easier for a Data Scientist perform the following
    tasks into the dev stage of their ML project deployment -

        1. Check for bias in data
        2. Check for bias in unmitigated model
        3. Check for bias in mitigated model
    """

    def __init__(
        self, protected_attribute_name: str, mitigation_algorithm: str = "reweighing"
    ):
        """__init__ constructor for FairAIChecker class

        Args:
            protected_attribute_name (str): Name of the protected-attribute column
            mitigation_algorithm (str, optional): Bias mitigation algorithm to be
            applied. Can take the following values -

                - 'reweighing' (default)
                - 'roc'
                - 'ceo'
                - 'eop'
        """
        self.protected_attribute_name = protected_attribute_name
        self.mitigation_algorithm = mitigation_algorithm
        self._configure_groups()

    def _configure_groups(self):
        """Configures unprivileged and privileged groups"""
        self.unprivileged_groups = [{self.protected_attribute_name: 0}]
        self.privileged_groups = [{self.protected_attribute_name: 1}]

    def _ml_model_training(self, X: pd.DataFrame, y, sample_weight=None):
        """Fits a ML model using X and y input data

        Args:
            X (pd.DataFrame): Input X
            y (_type_): Input target labels
            sample_weight (_type_, optional): Sample weights. Defaults to None.
        """

        self.LR = LogisticRegression(solver="liblinear", random_state=42)
        self.LR.fit(X, y, sample_weight=sample_weight)

    def _ml_model_prediction(self, X: pd.DataFrame):
        """Makes predictions on test-data using a trained ML model

        Args:
            X (pd.DataFrame): Input X

        Returns:
            _type_: y-prediction probabilities
        """
        y_pred_proba = self.LR.predict_proba(X)
        return y_pred_proba

    def fit_predict(self, X: pd.DataFrame, y, mitigator=None):
        """Fits the input data on a bias-mitigation algorithm, and
        computes the estimated bias scores of ML before and after mitigation

        Args:
            X (pd.DataFrame): X input data
            y (_type_): Target labels (binary)
            mitigator (_type_, optional): mitigator object. Defaults to None.
        """
        X_train, X_vt, y_train, y_vt = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_vt, y_vt, test_size=0.5, random_state=42
        )

        # Fit unmitigated model
        self._ml_model_training(X_train, y_train)
        y_val_pred_proba = self._ml_model_prediction(X_val)[:, 1].reshape(-1, 1)
        y_test_pred_proba = self._ml_model_prediction(X_test)[:, 1].reshape(-1, 1)
        y_test_pred = 1 * (y_test_pred_proba > 0.5)

        # Fit mitigated model
        # Pre-processing mitigation
        if self.mitigation_algorithm == "reweighing":
            print("Applying pre-processing bias-mitigation using Reweighing algorithm.")
            model = reweighing.ReweighingModel(self.protected_attribute_name)
            if mitigator is None:
                model.fit_model(X_train, y_train)
            weights = model.get_outputs(X_train, y_train, mitigator)

            self._ml_model_training(X_train, y_train, weights)
            y_test_pred_proba = self._ml_model_prediction(X_test)[:, 1].reshape(-1, 1)
            y_test_pred_mit = 1 * (y_test_pred_proba > 0.5)

        # Post-processing mitigation
        if self.mitigation_algorithm == "roc":
            print(
                "Applying post-processing bias-mitigation using Reject-Option-Classification (ROC) algorithm."
            )
            model = reject_option_classification.ROCModel(self.protected_attribute_name)
            if mitigator is None:
                model.fit_model(X_val, y_val, y_val_pred_proba)
            y_test_pred_mit = model.get_outputs(
                X_test, y_pred=y_test_pred, mitigator=mitigator
            )

        if self.mitigation_algorithm == "eop":
            print(
                "Applying post-processing bias-mitigation using Equality-of-Odds-Postprocessing (EOP) algorithm."
            )
            model = equality_of_odds.EqualityOfOddsModel(self.protected_attribute_name)
            if mitigator is None:
                model.fit_model(X_val, y_val, y_val_pred_proba)
            y_test_pred_mit = model.get_outputs(
                X_test, y_pred=y_test_pred, mitigator=mitigator
            )

        if self.mitigation_algorithm == "ceo":
            print(
                "Applying post-processing bias-mitigation using Calibrated-Equalized-Odds (CEO) algorithm."
            )
            model = calibrated_eq_odds.CEOModel(self.protected_attribute_name)
            if mitigator is None:
                model.fit_model(X_val, y_val, y_val_pred_proba)
            y_test_pred_mit = model.get_outputs(
                X_test, y_pred=y_test_pred, mitigator=mitigator
            )

        # generate bias-fairness report
        self.bias_scores = bias_fairness_report(
            X_test, y_test, y_test_pred, y_test_pred_mit, self.protected_attribute_name
        )

    def plot_fairness(self):
        """
        Plots fairness and performance metrics for unmitigated vs
        mitigated ML model, along with explanations on the metrics
        and the accuracy before and after mitigation.
        """
        privileged_group = 1

        bias_dict = copy.deepcopy(self.bias_scores)
        data_disparity = bias_dict["disparity"].pop("data")

        # create figure
        plt.style.use("fivethirtyeight")
        fig = plt.figure(figsize=(10, 5))

        # add subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.bar(
            bias_dict["disparity"].keys(),
            bias_dict["disparity"].values(),
            width=0.7,
            color=["lightcoral", "lightsteelblue"],
        )
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(
            bias_dict["error_rate_difference"].keys(),
            bias_dict["error_rate_difference"].values(),
            width=0.7,
            color=["lightcoral", "lightsteelblue"],
        )

        # set xlim and ylim
        ax1.set_xlim(-0.6, 1.5)
        ax1.set_ylim(0, 2)
        ax2.set_xlim(-0.6, 1.5)
        ax2.set_ylim(-1, 1)

        # set tick-labels
        ax1.xaxis.set_major_locator(mticker.FixedLocator(ax1.get_xticks()))
        ax1.yaxis.set_major_locator(mticker.FixedLocator(ax1.get_yticks()))
        ax2.xaxis.set_major_locator(mticker.FixedLocator(ax2.get_xticks()))
        ax2.yaxis.set_major_locator(mticker.FixedLocator(ax2.get_yticks()))
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
        ax1.text(s=round(bias_dict["disparity"]["unmitigated"], 2), x=-0.1, y=0.05)
        ax1.text(s=round(bias_dict["disparity"]["mitigated"], 2), x=0.9, y=0.05)
        ax2.text(
            s=round(bias_dict["error_rate_difference"]["unmitigated"], 2),
            x=-0.1,
            y=0.05,
        )
        ax2.text(
            s=round(bias_dict["error_rate_difference"]["mitigated"], 2), x=0.9, y=0.05
        )

        acc_unmit = round(bias_dict["accuracy"]["unmitigated"] * 100, 3)
        acc_mit = round(bias_dict["accuracy"]["mitigated"] * 100, 3)
        ax1.text(
            s=f'Sensitive Feature: "{self.protected_attribute_name}"',
            x=-0.5,
            y=2.7,
            verticalalignment="top",
            size=20,
            weight="bold",
        )
        ax1.text(
            s=f'Privileged Group: "{privileged_group}"',
            x=-0.5,
            y=2.5,
            verticalalignment="top",
        )
        ax1.text(
            s=f"Accuracy after mitigation changed from {acc_unmit}% to {acc_mit}%",
            x=-0.5,
            y=2.4,
            verticalalignment="top",
        )
        ax1.text(
            s=f"Bias mitigation algorithm applied: {self.mitigation_algorithm}",
            x=-0.5,
            y=2.3,
            verticalalignment="top",
        )
        ax1.text(
            s=f"Disparity in Input-Data: {data_disparity}",
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
            s="Error Rate Difference",
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
        The error rate difference gives the percentage of 
        transactions that were incorrectly scored by the 
        model. Formula - 
        
        Error Rate (ER) = (FP + FN) / (FP + FN + TP + TN)

        Error Rate Difference = ER_monitored - ER_reference

        The ideal value of this metric is 0. At 0, both
        groups have equal odds.
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
