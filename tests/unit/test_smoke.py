import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocessing_pipeline
from src.data.pipeline import create_preprocessing_pipeline


class TestSmokePreprocessing:

    def test_pipeline_runs_without_error(self, full_df):           
        result = preprocessing_pipeline(full_df.copy())
        assert result is not None

    def test_returns_expected_keys(self, full_df):                 
        result = preprocessing_pipeline(full_df.copy())
        expected_keys = {
            "X_train", "X_val", "X_test",
            "y_train", "y_val", "y_test",
            "X_train_scaled", "X_val_scaled", "X_test_scaled",
            "scaler", "df_encoded", "dataset_meta",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_consistent_shapes(self, full_df):                     
        result = preprocessing_pipeline(full_df.copy())
        assert len(result["X_train"]) == len(result["y_train"])
        assert len(result["X_val"])   == len(result["y_val"])
        assert len(result["X_test"])  == len(result["y_test"])

    def test_no_scaler_leakage(self, full_df):                     
        result = preprocessing_pipeline(full_df.copy())
        assert result["X_train_scaled"] is not result["X_test_scaled"]

    def test_rejects_empty_dataframe(self, empty_dataframe):
        with pytest.raises(Exception):
            preprocessing_pipeline(empty_dataframe)


class TestSmokeSklearnPipeline:

    def _prepare_split(self, sample_df):
        from src.data.preprocess import rename_target, convert_yes_no
        from src.config import TARGET

        df = convert_yes_no(sample_df.copy())
        df = rename_target(df)
        y = df[TARGET].astype(int)
        X = df.drop(columns=[TARGET])
        return train_test_split(X, y, test_size=0.4, random_state=42)

    def test_pipeline_fit_predict_runs(self, sample_df):
        from src.data.pipeline import create_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        X_train, X_test, y_train, y_test = self._prepare_split(sample_df) 
        pipeline = Pipeline([
            *create_preprocessing_pipeline().steps,
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert len(y_pred) == len(y_test)

    def test_predict_returns_binary(self, sample_df):
        from src.data.pipeline import create_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        X_train, X_test, y_train, y_test = self._prepare_split(sample_df)

        pipeline = Pipeline([
            *create_preprocessing_pipeline().steps,
            ("model", DummyClassifier(strategy="most_frequent")),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba_between_0_and_1(self, sample_df):
        from src.data.pipeline import create_preprocessing_pipeline
        from sklearn.pipeline import Pipeline

        X_train, X_test, y_train, y_test = self._prepare_split(sample_df)

        pipeline = Pipeline([
            *create_preprocessing_pipeline().steps,
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        assert np.all(y_proba >= 0.0)
        assert np.all(y_proba <= 1.0)