import os
import sys
import pandas as pd
import pickle
from ananke.graphs import ADMG
from ananke.estimation import CausalEffect
from sklearn import preprocessing
from tabulate import tabulate

class CARE_inf:
    def inference(self, bug_df, objective: str):
        with open('model/care.model', 'rb') as fp:
            care = pickle.load(fp)      
        vertices = care[0]
        di_edges = care[1]
        bi_edges = care[2]
        G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
        # G.draw(direction="LR")

        all_ace = []
        num_config = 18

        norm = preprocessing.scale(bug_df.values)
        df2 = pd.DataFrame(data=norm, columns=bug_df.columns)
        for config in range(num_config):
            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')
            ace_obj = CausalEffect(graph=G, treatment=vertices[config], outcome=objective)  # setting up the CausalEffect object
            sys.stdout = save_stdout
            ace = ace_obj.compute_effect(df2, "gformula")
            ace = abs(ace)
            all_ace.append(ace)
            debug = pd.DataFrame({'config':vertices[config], 'ace':[ace]})
            debug.to_csv('debug.csv', mode='a', index=False, header=False)

        debug = pd.read_csv('debug.csv', names=['config', 'ACE'])
        rank = debug.sort_values(by=['ACE'], ascending=False)
        # rank.to_csv('config_rank_ms.csv', index=False, header=False)
        # config_ace = pd.read_csv('config_rank_ms.csv', names=['config', 'ACE'])

        value = rank['ACE'].to_list()
        config = rank['config'].to_list()

        rank_1 = []
        rank_2 = []
        rank_3 = []
        bugs = [i for i, x in enumerate(value) if x == max(value)]
        for bug in range(len(bugs)): 
            root_causes = config[bug]
            rank_1.append(root_causes)

        del value[0:len(bugs)]
        del config[0:len(bugs)]
        bugs = [i for i, x in enumerate(value) if x == max(value)]
        for bug in range(len(bugs)): 
            root_causes = config[bug]
            rank_2.append(root_causes)

        del value[0:len(bugs)]
        del config[0:len(bugs)]
        bugs = [i for i, x in enumerate(value) if x == max(value)]
        for bug in range(len(bugs)): 
            root_causes = config[bug]
            rank_3.append(root_causes)

        ranks = [rank_1, rank_2, rank_3]
        os.remove('debug.csv')
        os.remove('trash')
        
        return ranks

    def get_predictions(self, ranks: list, config: str):
        if config in ranks:
            pred = 1
        else:
            pred = 0   
        return pred

    def _compare_ground_truth(self, config: str, ground_truth_data, prediction_data):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for (actual,pred) in zip(ground_truth_data[config], prediction_data[config]):
            if actual == 1 and pred == 1:
                TP += 1
            elif actual == 0 and pred == 0:
                TN += 1    
            elif actual == 1 and pred == 0: 
                FN += 1
            elif actual == 0 and pred == 1: 
                FP += 1
            else:
                TP = 0
                TN = 0
                FN = 0
                FP = 0   
        return TP, TN, FP, FN

    def _metrics(self, TP, TN, FP, FN):
        accuracy = round((TP+TN)/(TP+FP+TN+FN) * 100, 2)
        precision = round(TP/(TP+FP) * 100, 2)
        recall = round(TP/(TP+FN) * 100, 2)
        return accuracy, precision, recall

    def _compute_metrics(self, ground_truth_data, prediction_data):
        total_accuracy = []
        total_precision = []
        total_recal = []
        true_pos = []
        true_neg = []
        false_pos = []
        false_neg = []
        for config in prediction_data.columns:
            eval = CARE_inf()
            TP, TN, FP, FN = eval._compare_ground_truth(config, ground_truth_data, prediction_data)
            true_pos.append(TP)
            true_neg.append(TN)
            false_pos.append(FP)
            false_neg.append(FN)

        return true_pos, true_neg, false_pos, false_neg
    
    def eval_CARE(self, ground_truth_data, objective: str, platform: str):
        CI = CARE_inf()
        pred_df = pd.read_csv(f'data/{platform}/predictions_{objective}.csv').T
        tp, tn, fp, fn = CI._compute_metrics(ground_truth_data, pred_df)
        TP_sum = sum(tp)
        TN_sum = sum(tn)
        FP_sum = sum(fp)
        FN_sum = sum(fn) 
        accuracy = round((TP_sum+TN_sum)/(TP_sum+FP_sum+TN_sum+FN_sum) * 100, 2)
        precision = round(TP_sum/(TP_sum+FP_sum) * 100, 2)
        recall = round(TP_sum/(TP_sum+FN_sum) * 100, 2)
        F1_score = 2 * ((precision * recall) / (precision + recall))

        return accuracy, precision, recall, F1_score
    
    def misconfigs(self, bug_id, rank_1, rank_2, rank_3):
        bug_idx = bug_id
        r1 = rank_1
        r2 = rank_2
        r3 = rank_3
        for id in range(len(bug_idx)):
            r_1 = list(dict.fromkeys(r1[id]))
            r_2 = list(dict.fromkeys(r2[id]))
            r_3 = list(dict.fromkeys(r3[id]))
            for i in range(len(r_1)):
                df_result = pd.DataFrame({"bug_id":[bug_idx[id]],"misconfigs_rank1":[r_1[i]]})
                if not os.path.isfile('Rank_1.csv'):
                    df_result.to_csv('Rank_1.csv', mode='a', header=True, index=False)
                else:
                    df_result.to_csv('Rank_1.csv', mode='a', header=False, index=False)
            for i in range(len(r_2)):
                df_result = pd.DataFrame({"bug_id":[bug_idx[id]],"misconfigs_rank2":[r_2[i]]})
                if not os.path.isfile('Rank_2.csv'):
                    df_result.to_csv('Rank_2.csv', mode='a', header=True, index=False)
                else:
                    df_result.to_csv('Rank_2.csv', mode='a', header=False, index=False)
            for i in range(len(r_3)):
                df_result = pd.DataFrame({"bug_id":[bug_idx[id]],"misconfigs_rank3":[r_3[i]]})
                if not os.path.isfile('Rank_3.csv'):
                    df_result.to_csv('Rank_3.csv', mode='a', header=True, index=False)
                else:
                    df_result.to_csv('Rank_3.csv', mode='a', header=False, index=False)    

        care_result_r1 = pd.read_csv('Rank_1.csv')       
        care_result_r2 = pd.read_csv('Rank_2.csv')
        care_result_r3 = pd.read_csv('Rank_3.csv')

        return care_result_r1, care_result_r2, care_result_r3

    def reset_predictions(self, objective:str, platform:str):
        for file in range(3):
            if os.path.isfile(f'data/{platform}/predictions_{objective}.csv'):
                os.remove(f'data/{platform}/predictions_{objective}.csv')
            if os.path.isfile(f'Rank_{file+1}.csv'):
                os.remove(f'Rank_{file+1}.csv')

    def find_bug_files(self, path_to_dir, suffix=".csv"):
        filenames = os.listdir(path_to_dir)
        bug_files = [ filename for filename in filenames if filename.endswith( suffix ) ]
        bug_files.sort()
        return bug_files

    def run_care_inf(self, CI, bugfiles, columns, objective:str, platform:str):
        root_causes = {"bug_id":[], "misconfigs_r1":[],"misconfigs_r2":[], "misconfigs_r3":[]}
        for bug in bugfiles:
            bug_df = pd.read_csv(f'data/{platform}/bugs/{objective}/{bug}')
            ranks = CI.inference(bug_df, objective)
            for rank in range(len(ranks)):
                config_names = []
                config_preds = []
                for name in range(len(columns)):
                    config_name = f'{columns[name]}'
                    config_pred = CI.get_predictions(ranks[rank], columns[name])
                    config_names.append(config_name)
                    config_preds.append(config_pred)
                predictions = pd.DataFrame(data=[config_preds], columns=config_names)
                if not os.path.isfile(f'data/{platform}/predictions_{objective}_rank_{rank}.csv'):
                    predictions.to_csv(f'data/{platform}/predictions_{objective}_rank_{rank}.csv', mode='a', header=True, index=False)
                else:
                    predictions.to_csv(f'data/{platform}/predictions_{objective}_rank_{rank}.csv', mode='a', header=False, index=False)
            root_causes["bug_id"].append(bug)
            root_causes["misconfigs_r1"].append(ranks[0])
            root_causes["misconfigs_r2"].append(ranks[1])
            root_causes["misconfigs_r3"].append(ranks[2])

            rank1 = pd.read_csv(f'data/{platform}/predictions_{objective}_rank_0.csv')
            rank2 = pd.read_csv(f'data/{platform}/predictions_{objective}_rank_1.csv')
            rank3 = pd.read_csv(f'data/{platform}/predictions_{objective}_rank_2.csv')
            comb_pred = pd.concat([rank1, rank2, rank3], ignore_index=True) 

            if not os.path.isfile(f'data/{platform}/predictions_{objective}.csv'):
                (comb_pred.max().to_frame().T).to_csv(f'data/{platform}/predictions_{objective}.csv', mode='a', index=False)
            else:
                (comb_pred.max().to_frame().T).to_csv(f'data/{platform}/predictions_{objective}.csv', header=False, mode='a', index=False)
            for file in range(3):
                if os.path.isfile(f'data/{platform}/predictions_{objective}_rank_{file}.csv'):
                    os.remove(f'data/{platform}/predictions_{objective}_rank_{file}.csv')

        bug_idx = root_causes.get('bug_id')
        r1 = root_causes.get('misconfigs_r1')
        r2 = root_causes.get('misconfigs_r2')
        r3 = root_causes.get('misconfigs_r3')
        rank_1, rank_2, rank_3 = CI.misconfigs(bug_idx, r1, r2, r3)

        # print(tabulate(root_causes, headers='keys', tablefmt='rounded_outline'))  
        # print(tabulate(rank_1, headers='keys', tablefmt='rounded_outline'))
        # print(tabulate(rank_2, headers='keys', tablefmt='rounded_outline'))
        # print(tabulate(rank_3, headers='keys', tablefmt='rounded_outline'))
        
        return root_causes  