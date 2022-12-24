import os
import pandas as pd
import warnings
from tabulate import tabulate
from src.care_inference import CARE_inf

warnings.filterwarnings("ignore")


def husky_get_predictions_mission_success():
    CI = CARE_inf()
    CI.reset_predictions(objective="Mission_success", platform='husky')
    bugfiles = CI.find_bug_files("data/husky/bugs/Mission_success")
    df = pd.read_csv('data/husky/ground_truth_mission_success.csv')
    ground_truth = df.T
    columns = df.columns.to_list()
    root_causes = CI.run_care_inf(CI, bugfiles, columns, objective='Mission_success', platform='husky')
    accuracy, precision, recall, F1_score = CI.eval_CARE(ground_truth, objective='Mission_success', platform='husky')
    metrics = pd.DataFrame({"Accuracy":[accuracy], "Precision":[precision], "Recall":[recall]})
    # print(tabulate(metrics, headers='keys', tablefmt='outline', showindex=False))
    return accuracy, precision, recall, F1_score

def husky_get_predictions_energy():
    CI = CARE_inf()
    CI.reset_predictions(objective="Battery_percentage", platform='husky')
    bugfiles = CI.find_bug_files("data/husky/bugs/Battery_percentage")
    df = pd.read_csv('data/husky/ground_truth_energy.csv')
    ground_truth = df.T
    columns = df.columns.to_list()
    root_causes = CI.run_care_inf(CI, bugfiles, columns, objective='Battery_percentage', platform='husky')
    accuracy, precision, recall, F1_score = CI.eval_CARE(ground_truth, objective='Battery_percentage', platform='husky')
    metrics = pd.DataFrame({"Accuracy":[accuracy], "Precision":[precision], "Recall":[recall]})
    # print(tabulate(metrics, headers='keys', tablefmt='outline', showindex=False))
    return accuracy, precision, recall, F1_score

def turtlebot_get_predictions_mission_success():
    CI = CARE_inf()
    ground_truth = pd.read_csv('data/turtlebot3/ground_truth_mission_success.csv').T
    accuracy, precision, recall, F1_score = CI.eval_CARE(ground_truth, objective='Mission_success', platform='turtlebot3')
    metrics = pd.DataFrame({"Accuracy":[accuracy], "Precision":[precision], "Recall":[recall]})
    # print(tabulate(metrics, headers='keys', tablefmt='outline', showindex=False))
    return accuracy, precision, recall, F1_score

def turtlebot_get_predictions_energy():
    CI = CARE_inf()
    ground_truth = pd.read_csv('data/turtlebot3/ground_truth_energy.csv').T
    accuracy, precision, recall, F1_score = CI.eval_CARE(ground_truth, objective='Battery_percentage', platform='turtlebot3')
    metrics = pd.DataFrame({"Accuracy":[accuracy], "Precision":[precision], "Recall":[recall]})
    # print(tabulate(metrics, headers='keys', tablefmt='outline', showindex=False))
    return accuracy, precision, recall, F1_score

def husky():
    num_obj = 2
    ms_accuracy, ms_precision, ms_recall, ms_f1 = husky_get_predictions_mission_success()
    energy_accuracy, energy_precision, energy_recall, energy_f1 = husky_get_predictions_energy()
    total_accuracy = (ms_accuracy + energy_accuracy) / num_obj
    total_precision = (ms_precision + energy_precision) / num_obj
    total_recall = (ms_recall + energy_recall) / num_obj
    total_f1_score= (ms_f1 + energy_f1) / num_obj
    metrics = pd.DataFrame({"Platform":"Husky-sim","Accuracy":[total_accuracy], "Precision":[total_precision],
                            "Recall":[total_recall], "F1 Score":[total_f1_score]})
    return metrics

def turtlebot3():
    num_obj = 2
    ms_accuracy, ms_precision, ms_recall, ms_f1 = turtlebot_get_predictions_mission_success()
    energy_accuracy, energy_precision, energy_recall, energy_f1 = turtlebot_get_predictions_energy()
    total_accuracy = (ms_accuracy + energy_accuracy) / num_obj
    total_precision = (ms_precision + energy_precision) / num_obj
    total_recall = (ms_recall + energy_recall) / num_obj
    total_f1_score = (ms_f1 + energy_f1) / num_obj
    metrics = pd.DataFrame({"Platform":"Turtlebot3", "Accuracy":[total_accuracy], "Precision":[total_precision],
                            "Recall":[total_recall], "F1 Score":[total_f1_score]})
    return metrics


if __name__ == "__main__":
    husky_result = husky()
    turtlebot3_result = turtlebot3()
    df = pd.concat([husky_result, turtlebot3_result], ignore_index=True)
    print(tabulate(df, headers='keys', tablefmt='outline', showindex=False))