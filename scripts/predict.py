"""predict.py — Phase 6: Final predictions + single-row predict for FastAPI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from scripts.utils import load_object, report_metrics, OUTPUT_DIR
from scripts.preprocessing import (load_raw, step1_drop_duplicates, step2_drop_leakage,
    step3_drop_text, step4_fix_formatting, step5_handle_missing,
    step6_encode, step7_handle_outliers, step8_log_transform)
from scripts.feature_engineering import engineer_features

TARGET = "final_selling_price"

def load_pipeline():
    return (load_object("encoders.pkl"), load_object("outlier_caps.pkl"),
            load_object("log_cols.pkl"),  load_object("scaler.pkl"),
            load_object("selected_features.pkl"), load_object("best_model.pkl"))

def _predict(bundle, X):
    rf, lgbm, xgb = bundle["rf"], bundle["lgbm"], bundle["xgb"]
    p = (rf.predict(X) + lgbm.predict(X) + xgb.predict(X)) / 3
    return np.maximum(np.expm1(p), 0)

def run_predictions():
    print("\n=== PHASE 6 — FINAL PREDICTIONS ===")
    encoders,caps,log_cols,scaler,selected,bundle = load_pipeline()
    print(f"  Model: {load_object('best_model_name.pkl')}")

    df = load_raw()
    df = step1_drop_duplicates(df); df = step2_drop_leakage(df)
    df = step3_drop_text(df);       df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, _ = step6_encode(df,fit=False,encoders=encoders)
    df, _ = step7_handle_outliers(df,fit=False,caps=caps)
    df, _ = step8_log_transform(df,log_cols=log_cols)
    df    = engineer_features(df)

    X_all=df.drop(columns=[TARGET]); y_all=df[TARGET]
    _,X_te_r,_,y_test = train_test_split(X_all,y_all,test_size=0.2,random_state=42)
    X_te_s = pd.DataFrame(scaler.transform(X_te_r),columns=X_te_r.columns,index=X_te_r.index)
    X_test = X_te_s[selected]

    y_pred = _predict(bundle, X_test)

    print("\n  Performance:")
    metrics = report_metrics(y_test, y_pred, "Ensemble_Test")
    pct = np.abs(y_test.values - y_pred) / y_test.values * 100
    print(f"  Median % error : {np.median(pct):.1f}%")
    print(f"  Within 20%     : {(pct<=20).mean()*100:.1f}% of test samples")

    comp = pd.DataFrame({
        "True_Value": y_test.values, "Predicted": y_pred.round(0).astype(int),
        "Abs_Error":  np.abs(y_test.values-y_pred).round(0).astype(int),
        "Pct_Error":  pct.round(2)
    })
    comp.to_csv(os.path.join(OUTPUT_DIR,"predictions_comparison.csv"),index=False)
    return y_test, y_pred, metrics, comp

def predict_single(input_dict: dict) -> float:
    """Used by FastAPI: dict of raw features -> predicted EGP price."""
    encoders,caps,log_cols,scaler,selected,bundle = load_pipeline()
    df = pd.DataFrame([input_dict])
    df = step4_fix_formatting(df)
    df, _ = step6_encode(df,fit=False,encoders=encoders)
    df, _ = step7_handle_outliers(df,fit=False,caps=caps)
    df, _ = step8_log_transform(df,log_cols=log_cols)
    df    = engineer_features(df)
    df    = df.drop(columns=[TARGET],errors="ignore")
    X_sc  = pd.DataFrame(scaler.transform(df),columns=df.columns,index=df.index)
    return round(float(_predict(bundle, X_sc[selected])[0]), 2)

if __name__ == "__main__":
    run_predictions()