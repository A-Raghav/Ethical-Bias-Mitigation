import pandas as pd
from fairai.models.preprocessing import reweighing
from fairai.models.postprocessing import (
    reject_option_classification,
    calibrated_eq_odds,
    equality_of_odds,
)


class FairAIMitigator:
    """
    This class is a wrapper class around AIF-360 library used for bias and fairness.
    Contains methods that make it easier for a Data Scientist perform the following
    tasks into the prod stage of their ML project deployment -

        1. Fit a bias mitigation model based on algorithm selected
        2. Perform transformations/predictions
    """

    def __init__(
        self, protected_attribute_name: str, mitigation_algorithm: str = "reweighing"
    ):
        """__init__ constructor for FairAIMitigator class

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

    def fit(self, X_true: pd.DataFrame, y_true, y_pred_proba=None):
        """Fits a bias mitigation model depending on the algorithm selected by the
           user. Returns a mitigation-model object. Depending on the mitigation
           algorithm selected, the input can be different. Refer to the following
           information for more -

           A. Pre-Processing Algorithms:
                1. Reweighing:
                        Inputs:
                            * X_train
                            * y_train

            B. Post-Processing Algorithms:
                1. Reject-Option-Classification:
                2. Calibrated-Equality-of-Odds:
                        Inputs:
                            * X_val
                            * y_val
                            * y_val_pred_proba

        Args:
            X_true (pd.DatFrame): Training data
            y_true (_type_): Training data labels
            y_pred_proba (_type_, optional): Training data predictions. Defaults to None.

        Returns:
            _type_: mitigation model object
        """
        # Preprocessing algorithms
        if self.mitigation_algorithm == "reweighing":
            self.model = reweighing.ReweighingModel(self.protected_attribute_name)
            self.model.fit_model(X_true, y_true)

        # Postprocessing algorithms
        if self.mitigation_algorithm == "roc":
            self.model = reject_option_classification.ROCModel(
                self.protected_attribute_name
            )
            self.model.fit_model(X_true, y_true, y_pred_proba)

        if self.mitigation_algorithm == "ceo":
            self.model = calibrated_eq_odds.CEOModel(self.protected_attribute_name)
            self.model.fit_model(X_true, y_true, y_pred_proba)

        if self.mitigation_algorithm == "eop":
            self.model = equality_of_odds.EqualityOfOddsModel(
                self.protected_attribute_name
            )
            self.model.fit_model(X_true, y_true, y_pred_proba)

        self.mitigator = self.model.mitigator
        return self.mitigator

    def get_outputs(
        self,
        X: pd.DataFrame,
        y_true=None,
        y_pred=None,
        mitigator=None,
    ):
        """Transforms input data to get required output for a given bias mitigation
        strategy. Refer to the following information for more -

        A. Pre-Processing Algorithms:
            1. Reweighing:
                    Inputs:
                        * Fitted bias-mitigation model
                        * X_train, y_train
                    Outputs:
                        * sample_weight

        B. Post-Processing Algorithms
            1. Reject-Option-Classification
            2. Calibrated-Equalised-Odds
            3. Equality-of-Odds
                    Inputs:
                        * Fitted bias-mitigation model
                        * X_test
                        * y_test_pred
                    Outputs:
                        * y_test_pred_new

        A pre-existing/pickled mitigator model can also be used as input.

        Args:
            X (pd.DataFrame): Input data
            y_true (_type_, optional): y_true. Defaults to None.
            y_pred (_type_, optional): y-prediction probabilities. Defaults to None.
            mitigator (_type_, optional): fitted bias mitigation model

        Returns:
            _type_: Subjective, menioned above
        """
        # Preprocessing algorithms
        if self.mitigation_algorithm == "reweighing":
            sample_weights = self.model.get_outputs(X, y_true, mitigator)
            return sample_weights

        # Postprocessing algorithms
        if self.mitigation_algorithm == "roc":
            y_pred_new = self.model.get_outputs(X, y_pred, mitigator)

        if self.mitigation_algorithm == "ceo":
            y_pred_new = self.model.get_outputs(X, y_pred, mitigator)

        if self.mitigation_algorithm == "eop":
            y_pred_new = self.model.get_outputs(X, y_pred, mitigator)

        return y_pred_new
