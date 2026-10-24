"""
Chronos: Feature Engineering Module
Generates statistical temporal indicators and normalizes engine degradation telemetry.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .data_loader import INFORMATIVE_SENSORS

class TelemetryFeaturePipeline:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

    def _generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling statistics per engine to capture drift and mechanical degradation trends.
        """
        features_df = df.copy()
        
        # Calculate rolling mean and rolling standard deviation grouped by engine
        rolling_mean = (
            features_df.groupby("engine_id")[INFORMATIVE_SENSORS]
            .rolling(window=self.window_size, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_mean.columns = [f"{col}_roll_mean" for col in INFORMATIVE_SENSORS]
        
        rolling_std = (
            features_df.groupby("engine_id")[INFORMATIVE_SENSORS]
            .rolling(window=self.window_size, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )
        rolling_std.columns = [f"{col}_roll_std" for col in INFORMATIVE_SENSORS]
        
        # Merge raw sensors, rolling mean, and rolling std
        combined_df = pd.concat([
            features_df[["cycle"] + INFORMATIVE_SENSORS],
            rolling_mean,
            rolling_std
        ], axis=1)
        
        return combined_df

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on training telemetry features.
        """
        raw_features = self._generate_rolling_features(df)
        self.feature_names = list(raw_features.columns)
        self.scaler.fit(raw_features)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input telemetry into scaled feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform.")
        raw_features = self._generate_rolling_features(df)
        return self.scaler.transform(raw_features[self.feature_names])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def extract_test_latest_features(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the final recorded operational cycle for each test engine to predict current RUL.
        """
        transformed_matrix = self.transform(test_df)
        transformed_df = pd.DataFrame(transformed_matrix, columns=self.feature_names)
        transformed_df["engine_id"] = test_df["engine_id"].values
        
        # Select last cycle row for each engine
        last_cycle_indices = test_df.groupby("engine_id")["cycle"].idxmax()
        return transformed_matrix[last_cycle_indices]
