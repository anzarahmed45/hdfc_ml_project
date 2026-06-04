from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from sklearn.linear_model import LinearRegression


def get_logistic_regression():
    return LogisticRegression()


def get_decision_tree():
    return DecisionTreeClassifier(
        random_state=42
    )


def get_random_forest_classifier():
    return RandomForestClassifier(
        random_state=42
    )


def get_xgboost_classifier():
    return XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )


def get_linear_regression():
    return LinearRegression()


def get_random_forest_regressor():
    return RandomForestRegressor(
        random_state=42
    )


def get_gradient_boosting_regressor():
    return GradientBoostingRegressor(
        random_state=42
    )


def get_xgboost_regressor():
    return XGBRegressor(
        random_state=42
    )