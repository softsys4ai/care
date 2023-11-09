import pandas as pd
from src.baseline_CBI import DebuggingBaseline


outlier_data = pd.read_csv("data/husky/bugs/Battery_percentage/bug_1.csv")
config_columns = outlier_data.iloc[:, 0:18]
bug_value = 10 

CBI = DebuggingBaseline()

cbi_rootcause = CBI.measure_cbi_importance(data=outlier_data, 
                                            config_colmuns=config_columns,
                                            objective="Battery_percentage", 
                                            bug_val=bug_value,
                                            top_k=18)
print(cbi_rootcause)