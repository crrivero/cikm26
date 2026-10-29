import ast
import json
import math

import numpy as np
from scipy.stats import wilcoxon, chi2
import zipfile
from datetime import datetime
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def main():
    # TODO Run main.py and zip all the output files.
    zip_file_results = ''
    # TODO The folder where you want your results.
    pdf_folder_output = ''
    rel_type, eval_type = 'regular', 'Micro' # Micro or Macro

    # Skip the table of metrics if we don't have reversed.
    print_table, job_ids = True, ['0'] # ['0', '1'] or ['0']

    # Compute Kendall's W when we have many runs. Fancy is my attempt to plot extra stuff.
    print_w, print_fancy = False, True

    # This indicates whether to use weights (as normalization or weighted mean) or not.
    weight_type = 'Unweighted' # 'Unweighted', 'Normalization', 'Weighted'

    datasets = [
        'FB13', 'FB15K',
        'FB15K237', 'WN18', 'WN18RR', #'WN11',
        'BioKG', 'Hetionet', 'NELL-995', 'YAGO3-10'
    ]

    pretty_print_metrics = {
        'RR(rel=3)': '$RR_{m=3}$',
        'RR(rel=4)': '$RR_{m=4}$',
        'RR(rel=5)': '$RR_{m=5}$',

        'R(rel=3)@1': '$R@1_{m=3}$',
        'R(rel=4)@1': '$R@1_{m=4}$',
        'R(rel=5)@1': '$R@1_{m=5}$',

        'R(rel=3)@5': '$R@5_{m=3}$',
        'R(rel=4)@5': '$R@5_{m=4}$',
        'R(rel=5)@5': '$R@5_{m=5}$',

        'R(rel=3)@10': '$R@10_{m=3}$',
        'R(rel=4)@10': '$R@10_{m=4}$',
        'R(rel=5)@10': '$R@10_{m=5}$',

        'Bpref(rel=3)': '$Bpref_{m=3}$',
        'Bpref(rel=4)': '$Bpref_{m=4}$',
        'Bpref(rel=5)': '$Bpref_{m=5}$',

        'RBP(rel=3)': '$RBP_{m=3}$',
        'RBP(rel=4)': '$RBP_{m=4}$',
        'RBP(rel=5)': '$RBP_{m=5}$',

        'nDCG@10': '$nDCG@10$',
        'SDCG(min_rel=1,max_rel=5)@10': '$SDCG@10$',
        'INSQ(min_rel=1,max_rel=5)': '$INSQ$',
        'BPM(min_rel=1,max_rel=5)@10': '$BPM@10$',
    }

    pretty_print_models = {
        'boxe': 'Bo', 'complex': 'Co', 'hake_both': 'HA', 'hole': 'Ho', 'quate': 'Qu', 'rotate': 'Rt',
        'rotpro': 'RP', 'toruse_eL2': 'To', 'transe': 'Tr', 'tucker': 'Tu'
    }

    alpha = .05

    global_metrics, global_models = None, None
    data_by_dataset_metric = {}

    with (zipfile.ZipFile(zip_file_results, 'r') as zf):
        # Get all the files in the file.
        for filename in zf.namelist():
            print('Going for', filename, '--', datetime.now())

            weights = None
            values_by_model_metric = {}
            with zf.open(filename, 'r') as file_in_zip:
                for line in file_in_zip:
                    line = line.decode('utf-8')

                    if line.startswith('Metadata: '):
                        # It is a well-formed document.
                        json_metadata = json.loads(line.replace('Metadata: ', ''))

                        # We only care about "relevance_type": "regular", "evaluation_type": "micro"
                        if 'dataset' in json_metadata:
                            dataset = json_metadata['dataset']

                            if json_metadata['relevance_type'] != rel_type or \
                                    json_metadata['evaluation_type'] != eval_type or \
                                    dataset not in datasets:
                                break

                            if dataset not in data_by_dataset_metric:
                                data_by_dataset_metric[dataset] = {}

                        if 'metrics' in json_metadata:
                            metrics = json_metadata['metrics']
                            models = json_metadata['models']

                            if global_metrics is None:
                                global_metrics = metrics
                            elif global_metrics != metrics:
                                print('Metrics were different!', metrics, 'vs.', global_metrics)
                                return

                            if global_models is None:
                                global_models = models
                            elif global_models != models:
                                print('Models were different!', models, 'vs.', global_models)
                                return

                    # We wish to process results only.
                    if line.startswith('D: '):
                        list_tuples = ast.literal_eval(line.replace('D: ', ''))

                        for model, metric, value in list_tuples:
                            if model not in values_by_model_metric:
                                values_by_model_metric[model] = {}

                            if metric not in values_by_model_metric[model]:
                                values_by_model_metric[model][metric] = []

                            values_by_model_metric[model][metric].append(value)

                    if line.startswith('Weights: '):
                        weights = json.loads('{"w": ' + line.replace('Weights: ', '') + '}')['w']

            # We compute ranks and the stats summary (mean, std).
            for metric in global_metrics:
                model_summary = {}

                if metric not in data_by_dataset_metric[dataset]:
                    data_by_dataset_metric[dataset][metric] = {}

                for model in global_models:
                    measurements = values_by_model_metric[model][metric]

                    if weight_type == 'Unweighted':
                        mean = np.average(measurements)
                        std = np.std(measurements)

                    model_summary[model] = (mean, std)

                data_by_dataset_metric[dataset][metric]['summary'] = model_summary

                # Sort models by mean (reverse order to get max first).
                sorted_models = list(global_models)
                sorted_models.sort(key=lambda current_m: model_summary[current_m][0], reverse=True)

                # Compare side by side using Wilcoxon test.
                positions = {}
                data_by_dataset_metric[dataset][metric]['positions'] = positions

                # sorted[i-1] vs. sorted[i]: if Wilcoxon test is true (they are diff. at alpha), we keep the positions.
                #   If test is false, sorted[i-1] and sorted[i] are tied.
                current_rank, ties = 1, 0
                for i in range(1, len(sorted_models)):
                    imomodel, imodel = sorted_models[i-1], sorted_models[i]
                    wresult = wilcoxon(x=values_by_model_metric[imomodel][metric], y=values_by_model_metric[imodel][metric])

                    statistic, pvalue = wresult[0], wresult[1]

                    if pvalue < alpha:
                        if current_rank not in positions:
                            ties = 0
                            positions[current_rank] = [(imomodel, model_summary[imomodel])]
                        else:
                            ties += 1
                            positions[current_rank].append((imomodel, model_summary[imomodel]))

                        current_rank += 1 + ties

                        # If we are in the last position, imodel must be added as well!
                        if i == len(sorted_models) - 1:
                            positions[current_rank] = [(imodel, model_summary[imodel])]
                    else:
                        if current_rank not in positions:
                            ties = 0
                            positions[current_rank] = [(imomodel, model_summary[imomodel])]
                        else:
                            ties += 1
                            positions[current_rank].append((imomodel, model_summary[imomodel]))

                        # If we are in the last position, imodel must be added as well!
                        if i == len(sorted_models) - 1:
                            positions[current_rank].append((imodel, model_summary[imodel]))

    if print_table:
        # Only print for 0,1 jobs (original splits, regular and reversed relevance values).
        metrics_to_print = list(pretty_print_metrics.keys())

        for dataset in datasets:
            to_print = ''

            for m_idx, metric in enumerate(metrics_to_print):
                if m_idx == 0:
                    to_print += f'\\multirow{{{len(metrics_to_print)}}}{{7pt}}{{\\rotatebox{{90}}{{{dataset}}}}} & '
                else:
                    to_print += '~ & '

                to_print += pretty_print_metrics[metric] + ' & '

                positions = data_by_dataset_metric[dataset][metric]['positions']
                rank_list = list(positions.keys())

                total = 0
                for rank in rank_list:
                    total += len(positions[rank])

                # Let's print the winner!
                tuple = positions[rank_list[0]]
                to_print += f' {tuple[0][1][0]:.3f}$\\pm${tuple[0][1][1]:.3f} & $'

                for i, rank in enumerate(rank_list):
                    for j, (m, _) in enumerate(positions[rank]):
                        to_print += pretty_print_models[m]
                        total -= 1

                        if total > 0:
                            if j == len(positions[rank]) - 1:
                                to_print += '>'
                            else:
                                to_print += '='

                to_print += '$ \\\\ \n'

            to_print += '\\hline'

            print(to_print)

    # if print_w:
    #     # For even job ids study how the model ranks are preserved. Some kind of error would work.
    #     # m, n
    #     total_jobs, total_models = 26, 10
    #
    #     # Implementing Kendall's W: https://en.wikipedia.org/wiki/Kendall%27s_W#Steps_of_Kendall's_W
    #     all_ws, all_pvalues = [], []
    #     for dataset in datasets:
    #         for metric in pretty_print_metrics:
    #             # The job id will be the judge.
    #
    #             # This is rij, the rank of object i given by judge j.
    #             rij = {}
    #             # This is the correction for ties.
    #             tj = []
    #
    #             for j in range(0, total_jobs):
    #                 for rank, models in data_by_dataset_jobid_metric[dataset][str(j*2)][metric]['positions'].items():
    #                     # Handle ties: https://en.wikipedia.org/wiki/Kendall%27s_W#Correction_for_Ties
    #                     if len(models) > 1:
    #                         t_jk = len(models)
    #                         tj.append(t_jk**3 - t_jk)
    #
    #                     for m, _ in models:
    #                         if m not in rij:
    #                             rij[m] = []
    #
    #                         rij[m].append((2 * rank + (len(models)-1)) / 2)
    #
    #             # Formula: https://en.wikipedia.org/wiki/Kendall%27s_W#Correction_for_Ties
    #             ri = []
    #             for m in rij:
    #                 ri.append(sum(r for r in rij[m]))
    #
    #             w_num = (12 * sum(r ** 2 for r in ri) -
    #                      3 * total_jobs**2 * total_models * (total_models + 1)**2)
    #
    #             w_den = total_jobs**2 * total_models * (total_models**2 - 1) - total_jobs * sum(t for t in tj)
    #
    #             # Final value for this particular metric.
    #             if w_den > 0:
    #                 w = w_num / w_den
    #
    #                 # Get p-value: https://en.wikipedia.org/wiki/Kendall%27s_W#Significance_Tests
    #                 chi2_stat = total_jobs * (total_models - 1) * w
    #                 p_value = chi2.sf(chi2_stat, df=total_models - 1)
    #             else:
    #                 # All are tied (it happens for some metrics).
    #                 w = 1.0
    #                 p_value = 0.0
    #
    #             all_ws.append(w)
    #             all_pvalues.append(p_value)
    #
    #             print(f'Dataset: {dataset}; Metric: {metric}; W: {w}; p-value: {p_value}')
    #
    #     def print_five_number_summary(values):
    #         # Calculate the five-number summary
    #         minimum = np.min(values)
    #         q1 = np.quantile(values, 0.25)
    #         median = np.median(values)
    #         q3 = np.quantile(values, 0.75)
    #         maximum = np.max(values)
    #
    #         print(f"Minimum: {minimum}, Q1: {q1}, Median: {median}, Q3: {q3}, Maximum: {maximum}")
    #
    #     print('\n\n')
    #     print('Five-number summary for W values: ')
    #     print_five_number_summary(np.array(all_ws))
    #
    #     print('\n\n')
    #     print('Five-number summary for p-values: ')
    #     print_five_number_summary(np.array(all_pvalues))

    # if print_fancy:
    #     # For each dataset and each job type:
    #     for dataset in datasets:
    #         cd_metrics = {
    #             'binary':[
    #                 'Bpref(rel=4)', 'Bpref(rel=5)',  'RBP(rel=4)', 'RBP(rel=5)', 'Rprec(rel=4)', 'Rprec(rel=5)',
    #                 'RR(rel=4)', 'RR(rel=5)', 'SetP(rel=4)', 'SetP(rel=5)', 'SetR(rel=4)', 'SetR(rel=5)',
    #                 'P(rel=4)@1', 'P(rel=5)@1', 'P(rel=4)@5', 'P(rel=5)@5', 'P(rel=4)@10', 'P(rel=5)@10',
    #                 'R(rel=4)@1', 'R(rel=5)@1', 'R(rel=4)@5', 'R(rel=5)@5', 'R(rel=4)@10', 'R(rel=5)@10',
    #                 'Success(rel=4)@10', 'Success(rel=5)@10'],
    #             'non_binary': [
    #                 'INSQ(min_rel=1,max_rel=5)', 'INST(min_rel=1,max_rel=5)',
    #                 'NERR10(min_rel=1,max_rel=5)', 'NERR11(min_rel=1,max_rel=5)',
    #                 'BPM(min_rel=1,max_rel=5)@10', 'nDCG@10', 'NERR8(min_rel=1,max_rel=5)@10',
    #                 'NERR9(min_rel=1,max_rel=5)@10', 'SDCG(min_rel=1,max_rel=5)@10',
    #             ]
    #         }
    #
    #         for metric_type in cd_metrics:
    #             metric_column, model_column, model_means = [], [], []
    #
    #             # Collect model means.
    #             for metric in cd_metrics[metric_type]:
    #                 for model in model_list:
    #                     (mean, std_dev) = data_by_dataset_jobid_metric[dataset][jobid][metric]['summary'][model]
    #
    #                     metric_column.append(metric)
    #                     model_column.append(model)
    #                     model_means.append(mean)
    #
    #             data = pd.DataFrame({'Metric': metric_column, 'Model': model_column, 'Mean': model_means})
    #
    #             # https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html#critical-difference-diagrams
    #
    #             # Get the average (percentile) ranks:
    #             avg_rank = data.groupby('Metric').Mean.rank(pct=True).groupby(data.Model).mean()
    #
    #             # Get the results of a post hoc Conover test:
    #             test_results = sp.posthoc_conover_friedman(
    #                 data, melted=True, block_col='Metric', block_id_col='Metric', group_col='Model', y_col='Mean')
    #
    #             # Visualize (optional):
    #             (ax, _) = sp.sign_plot(test_results)
    #             fig = ax.get_figure()
    #
    #             # plt.title(f'SignificancePlot_{dataset}_{rubric_type}_{weight_type}_{eval_type}_{metric_type}')
    #             with (PdfPages(
    #                     pdf_folder_output +
    #                     f'SignificancePlot_{dataset}_{rubric_type}_{weight_type}_{eval_type}_{metric_type}.pdf')
    #             as pdf):
    #                 pdf.savefig(fig, bbox_inches='tight')
    #
    #             plt.close(fig)
    #
    #             # Plot the results:
    #             fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
    #             sp.critical_difference_diagram(avg_rank, test_results)
    #
    #             # plt.title(f'CD_{dataset}_{rubric_type}_{weight_type}_{eval_type}_{metric_type}')
    #             with PdfPages(
    #                     pdf_folder_output +
    #                     f'CD_{dataset}_{rubric_type}_{weight_type}_{eval_type}_{metric_type}.pdf') as pdf:
    #                 pdf.savefig(fig, bbox_inches='tight')
    #
    #             plt.close(fig)
    #
    #     # For each metric, let's plot heatmaps.
    #     for metric in metrics:
    #         for jobid in ['0', '1']:
    #             print(f'Metric: {metric}; Job: {jobid}')
    #             rubric_type = 'regular' if jobid == '0' else 'reversed'
    #
    #             dataset_column, model_column, model_means = [], [], []
    #
    #             # Collect model means.
    #             for dataset in datasets:
    #                 for model in model_list:
    #                     (mean, std_dev) = data_by_dataset_jobid_metric[dataset][jobid][metric]['summary'][model]
    #
    #                     dataset_column.append(dataset)
    #                     model_column.append(model)
    #                     model_means.append(mean)
    #
    #             data = pd.DataFrame({'Dataset': dataset_column, 'Model': model_column, 'Mean': model_means})
    #
    #             heatmap_df = data.pivot(index='Model', columns='Dataset', values='Mean')
    #
    #             if weight_type != 'Unweighted':
    #                 colormap, current_vmin, current_vmax = 'coolwarm', -.1, .1
    #             else:
    #                 colormap, current_vmin, current_vmax = 'viridis', 0, .5 # By default
    #                 if str(metric).startswith('BPM') or str(metric).startswith('SDCG'):
    #                     current_vmax = 0.75
    #                 if str(metric).startswith('nDCG'):
    #                     current_vmax = 0.85
    #                 if str(metric).startswith('R('):
    #                     current_vmax = 0.85
    #                 if str(metric).startswith('Success'):
    #                     current_vmax = 0.85
    #                 if str(metric).startswith('SetR'):
    #                     current_vmax = 1.00
    #                 if str(metric).startswith('RR('):
    #                     current_vmax = 0.75
    #
    #             ax = sns.heatmap(heatmap_df,
    #                              #annot=True,  # show values
    #                              fmt=".3f",  # formatting
    #                              linewidths=0.5,
    #                              square=True,
    #                              cmap=colormap,
    #                              vmin=current_vmin,
    #                              vmax=current_vmax,
    #                              )
    #
    #             fig = ax.get_figure()
    #
    #             # plt.title(f'HeatMap_{metric}_{rubric_type}_{weight_type}_{eval_type}')
    #             with PdfPages(
    #                     pdf_folder_output + f'HeatMap_{metric}_{rubric_type}_{weight_type}_{eval_type}.pdf') as pdf:
    #                 pdf.savefig(fig, bbox_inches='tight')
    #
    #             plt.close(fig)



if __name__ == "__main__":
    main()