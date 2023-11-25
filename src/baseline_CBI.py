import math
import pandas as pd
import operator
from typing import Dict

class DebuggingBaseline:
    def __init__(self):
        print("initializing DebuggingBaselines class")

    def measure_cbi_importance(self, data: pd.DataFrame, config_columns: pd.DataFrame.columns, 
                               objective: str, bug_val: float, top_k: int) -> Dict:
        """This function is used to measure importance"""
        Importance = {}

        adjustment = 3
        threshold = 0.7 * bug_val
        for col in config_columns:
            unique_values = data[col].unique()
            for val in unique_values:
                # compute F (P_true)
                P_true_df = data.loc[data[col] == val]
                F_P_true_df = P_true_df.loc[P_true_df[objective] > threshold]
                F_P_true = len(F_P_true_df) + adjustment 

                # compute S (P_true)
                S_P_true_df = P_true_df.loc[P_true_df[objective] < threshold]
                S_P_true = len(S_P_true_df) + adjustment

                # compute F (P_observed)
                F_P_observed_df = data.loc[data[objective] > threshold]
                F_P_observed = len(F_P_observed_df)
                # compute S (P_observed)
                S_P_observed_df = data.loc[data[objective] < threshold]
                S_P_observed = len(S_P_observed_df)

                # compute Failure
                Failure_P = F_P_true / (S_P_true + F_P_true) if (S_P_true + F_P_true) != 0 else 0

                # compute Context
                Context_P = F_P_observed / (S_P_observed + F_P_observed) if (S_P_observed + F_P_observed) != 0 else 0

                # compute Increase
                Increase_P = Failure_P - Context_P + 0.1
                        
                # compute Importance
                if F_P_true > 0 and F_P_observed > 0:
                    Importance_P = 2 / (1 / Increase_P) + (1 / (math.log(F_P_true) / math.log(F_P_observed)))
                else:
                    Importance_P = 0

                Importance[f"{col}_{val}"] = Importance_P
                
        return sorted(Importance.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
