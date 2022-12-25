import os
import pickle
import pandas as pd
from src.causal_model import rcausal
from ananke.graphs import ADMG
from networkx import DiGraph

if os.path.isfile("result/care_ace_result.csv"):
    os.remove("result/care_ace_result.csv")

def run_care(rcausal, df, tabu_edges_remove, columns, objectives, alpha, NUM_PATHS):
        fci_edges = rcausal.care_fci(df, tabu_edges_remove, alpha)
        edges = []
        # resolve notears_edges and fci_edges and update
        di_edges, bi_edges = rcausal.resolve_edges(edges, fci_edges, columns,
                                            tabu_edges_remove, objectives, NUM_PATHS)
        G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)
        return G, di_edges, bi_edges


if __name__ == '__main__':
    alpha = 0.8     # use 0.8
    NUM_PATHS = 1
    
    # nodes for causal graph
    df = pd.read_csv('observational_data/Evaluation_results_2022_09_04-04_09_49_PM_3L.csv')
    columns = df.columns
    opt = ['Cost_scaling_factor_global','Update_frequency_global','Publish_frequency_global',
           'Transform_tolerance_global','Combination_method_global', 'Cost_scaling_factor_local',
           'Inflation_radius_local','Update_frequency_local', 'Publish_frequency_local',
           'Combination_method_local','Transform_tolerance_local', 'Path_distance_bias',
           'Goal_distance_bias','Occdist_scale', 'Stop_time_buffer',
           'yaw_goal_tolerance', 'xy_goal_tolerance','min_vel_x']
    options = df[opt]
    met = ['RNS','Traveled_distance','Mission_time', 'DWA_new_plan',
           'recovery_executed', 'DWA_failed', 'Error_rotating_goal']
    metrics = df[met]  
    obj = ['Battery_percentage','Mission_success']
    objectives = df[obj]

    # initialize causal model object
    CM = rcausal(columns)
    g = DiGraph()
    g.add_nodes_from(columns)

    # edge constraints
    tabu_edges_remove = CM.get_tabu_edges_care_remove(options, objectives, metrics)
    G, di_edges, bi_edges = run_care(CM, df, tabu_edges_remove,
                                             columns, objectives, alpha, NUM_PATHS)
    
    # saving the model
    care = [columns, di_edges, bi_edges]
    with open('model/care.model', 'wb') as fp:
        pickle.dump(care, fp)

    g.add_edges_from(di_edges + bi_edges)    

    paths = CM.get_causal_paths(columns, di_edges, bi_edges,
                                objectives)
    # compute causal paths
    for key, val in paths.items():
        if len(paths[key]) > NUM_PATHS:
            s = CM.compute_path_causal_effect(df, paths[key], G, NUM_PATHS)
        else:
            paths = paths[objectives[0]]