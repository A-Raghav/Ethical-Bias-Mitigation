from aif360.algorithms.preprocessing import Reweighing

from fairai.models import BaseMitigator


class ReweighingModel(BaseMitigator):
    def __init__(self, protected_attribute_name: str):
        self.protected_attribute_name = protected_attribute_name

    def fit_model(self, X, y):
        dataset = self.create_dataset(X, y)

        privileged_groups = [{self.protected_attribute_name: 1.0}]
        unprivileged_groups = [{self.protected_attribute_name: 0.0}]

        self.mitigator = Reweighing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
        )
        self.mitigator.fit(dataset)

    def get_outputs(self, X, y_true, mitigator=None):
        if mitigator is not None:
            self.mitigator = mitigator
        dataset = self.create_dataset(X, y_true)
        dataset_transf = self.mitigator.transform(dataset)
        sample_weights = dataset_transf.instance_weights
        return sample_weights
