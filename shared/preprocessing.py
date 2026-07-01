"""Shared sklearn preprocessing pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(numeric_features, categorical_features, sparse_output=None):
    """Build a ColumnTransformer with standard numeric and categorical pipelines.

    Parameters
    ----------
    numeric_features : list[str]
        Column names to impute with median strategy.
    categorical_features : list[str]
        Column names to impute (most_frequent) then one-hot encode.
    sparse_output : bool | None
        If set, pass ``sparse_output`` to OneHotEncoder.  When None the
        encoder uses its default (sparse for older sklearn, dense for newer).

    Returns
    -------
    ColumnTransformer
    """
    onehot_kwargs = {"handle_unknown": "ignore"}
    if sparse_output is not None:
        onehot_kwargs["sparse_output"] = sparse_output

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(**onehot_kwargs)),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers)
