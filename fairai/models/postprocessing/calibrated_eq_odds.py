from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)

from fairai.models import BaseMitigator


class CEOModel(BaseMitigator):
    def __init__(self, protected_attribute_name: str):
        self.protected_attribute_name = protected_attribute_name

    def fit_model(self, X_true, y_true, y_pred_proba):
        dataset = self.create_dataset(X_true, y_true)
        dataset_pred = dataset.copy()
        dataset_pred.labels = 1 * (y_pred_proba > 0.5)
        dataset_pred.scores = y_pred_proba

        privileged_groups = [{self.protected_attribute_name: 1.0}]
        unprivileged_groups = [{self.protected_attribute_name: 0.0}]

        self.mitigator = CalibratedEqOddsPostprocessing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            cost_constraint="fnr",
            seed=42,
        )
        self.mitigator.fit(dataset, dataset_pred)

    def get_outputs(self, X, y_pred, mitigator=None):
        if mitigator is not None:
            self.mitigator = mitigator
        dataset = self.create_dataset(X, y_pred)
        y_pred_mit = self.mitigator.predict(dataset).labels
        return y_pred_mit
