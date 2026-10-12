"""PaySim feature definitions shared by training and inference."""
import pandas as pd

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Inputs known when an authorization decision is made.
PRE_AUTH_RAW_COLS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
PRE_AUTH_NUM_COLS = ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]

# Retrospective regime used only to quantify post-authorization leakage.
POST_AUTH_RAW_COLS = PRE_AUTH_RAW_COLS + ["newbalanceOrig", "newbalanceDest"]
POST_AUTH_NUM_COLS = PRE_AUTH_NUM_COLS + [
    "newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"
]

# Backwards-compatible names for older notebooks and model artifacts.
RAW_COLS = POST_AUTH_RAW_COLS
NUM_COLS = POST_AUTH_NUM_COLS
CAT_COLS = ["type"]


def add_pre_auth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only fields available before transaction authorization."""
    return df[PRE_AUTH_RAW_COLS].copy()


def add_post_auth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add balance discrepancy signals that require completed transaction data."""
    result = df[POST_AUTH_RAW_COLS].copy()
    result["errorBalanceOrig"] = result["newbalanceOrig"] + result["amount"] - result["oldbalanceOrg"]
    result["errorBalanceDest"] = result["oldbalanceDest"] + result["amount"] - result["newbalanceDest"]
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy alias for post-authorization feature engineering."""
    return add_post_auth_features(df)
