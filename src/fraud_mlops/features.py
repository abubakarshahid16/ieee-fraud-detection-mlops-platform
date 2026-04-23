from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.mapping_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []

    def _to_frame(self, x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        return pd.DataFrame(x, columns=self.columns_ or None)

    def fit(self, x: pd.DataFrame, y=None):
        frame = self._to_frame(x)
        self.columns_ = list(frame.columns)
        self.mapping_ = {
            column: frame[column].astype(str).value_counts(normalize=True).to_dict()
            for column in self.columns_
        }
        return self

    def transform(self, x: pd.DataFrame):
        frame = self._to_frame(x)
        transformed = pd.DataFrame(index=frame.index)
        for column in self.columns_:
            transformed[f"{column}_freq"] = (
                frame[column].astype(str).map(self.mapping_[column]).fillna(0.0)
            )
        return transformed

    def get_feature_names_out(self, input_features=None):
        base = input_features if input_features is not None else self.columns_
        return [f"{column}_freq" for column in base]


@dataclass
class FeatureArtifacts:
    pipeline: ColumnTransformer
    selected_feature_names: list[str] = field(default_factory=list)


def reduce_feature_space(
    df: pd.DataFrame,
    max_missing_ratio: float,
    max_numeric_features: int,
    max_categorical_features: int,
):
    missing_ratio = df.isna().mean()
    retained = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    filtered = df[retained].copy()

    numeric_columns = filtered.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = filtered.select_dtypes(exclude=["number"]).columns.tolist()

    if len(numeric_columns) > max_numeric_features:
        numeric_rank = filtered[numeric_columns].nunique(dropna=False).sort_values(ascending=False)
        keep_numeric = numeric_rank.head(max_numeric_features).index.tolist()
    else:
        keep_numeric = numeric_columns

    if len(categorical_columns) > max_categorical_features:
        categorical_rank = (
            filtered[categorical_columns].nunique(dropna=False).sort_values(ascending=False)
        )
        keep_categorical = categorical_rank.head(max_categorical_features).index.tolist()
    else:
        keep_categorical = categorical_columns

    selected_columns = keep_numeric + keep_categorical
    return filtered[selected_columns].copy()


def split_feature_types(df: pd.DataFrame, top_high_cardinality: int):
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    high_cardinality = sorted(
        categorical_columns,
        key=lambda column: df[column].nunique(dropna=False),
        reverse=True,
    )[:top_high_cardinality]
    low_cardinality = [column for column in categorical_columns if column not in high_cardinality]
    return numeric_columns, low_cardinality, high_cardinality


def build_feature_pipeline(df: pd.DataFrame, top_high_cardinality: int) -> FeatureArtifacts:
    numeric_columns, low_cardinality, high_cardinality = split_feature_types(df, top_high_cardinality)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_missing_pipeline = Pipeline(
        steps=[
            ("indicator", MissingIndicator(features="all")),
        ]
    )
    low_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    high_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", FrequencyEncoder()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("num_missing", numeric_missing_pipeline, numeric_columns),
            ("low_cat", low_card_pipeline, low_cardinality),
            ("high_cat", high_card_pipeline, high_cardinality),
        ],
        remainder="drop",
    )
    return FeatureArtifacts(pipeline=transformer)


def fit_transform_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_high_cardinality: int,
    max_missing_ratio: float,
    max_numeric_features: int,
    max_categorical_features: int,
):
    train_reduced = reduce_feature_space(
        train_df,
        max_missing_ratio=max_missing_ratio,
        max_numeric_features=max_numeric_features,
        max_categorical_features=max_categorical_features,
    )
    test_reduced = test_df.reindex(columns=train_reduced.columns, fill_value=None)
    artifacts = build_feature_pipeline(train_reduced, top_high_cardinality)
    x_train = artifacts.pipeline.fit_transform(train_reduced)
    x_test = artifacts.pipeline.transform(test_reduced)
    return x_train, x_test, artifacts, train_reduced.columns.tolist()


def select_features(x_train, y_train, x_test):
    selector_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    selector = SelectFromModel(selector_model, threshold="median")
    selector.fit(x_train, y_train)
    return selector.transform(x_train), selector.transform(x_test), selector
