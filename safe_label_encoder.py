# safe_label_encoder.py

import numpy as np
from sklearn.preprocessing import LabelEncoder

class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        is_scalar = False
        if np.isscalar(y):
            y = [y]
            is_scalar = True

        known_classes = set(self.classes_)
        base_transform = super(SafeLabelEncoder, self).transform

        transformed = [
            base_transform([label])[0] if label in known_classes else -1
            for label in y
        ]
        return transformed[0] if is_scalar else np.array(transformed)


    def inverse_transform(self, y):
        result = []
        for val in y:
            if val == -1:
                result.append("Unknown")
            else:
                result.append(super().inverse_transform([val])[0])
        return result
