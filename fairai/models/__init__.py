import pandas as pd
import numpy as np

from aif360.datasets import StandardDataset


class BaseMitigator:
    def create_dataset(self, X: pd.DataFrame, y) -> StandardDataset:
        if isinstance(y, np.ndarray):
            y = pd.Series(y.flatten(), index=X.index, name="class")
        return StandardDataset(
            df=pd.concat([X, y], axis=1),
            label_name="class",
            favorable_classes=[1],
            protected_attribute_names=[self.protected_attribute_name],
            privileged_classes=[[1]],
        )
