import os.path

import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json


def main():
    # TODO Your results after running ProcessMeasurements.py.
    folder_input = ''
    eval_types = ['micro_all', 'macro_all', 'macro_singleton', 'macro_non-singleton']

    datasets = [
        'FB13', 'FB15K', 'FB15K-237',
        'WN18', 'WN18RR', #'WN11',
        'BioKG', 'Hetionet', 'NELL-995', 'YAGO3-10'
    ]

    pretty_print_metrics = {
        'RR(rel=5)': "$RR$",
        'Bpref(rel=5)': "$Bpref$",
        'Rprec(rel=5)': "$Rprec$",
        'AP(rel=5)': "$AP$",
        'Success(rel=5)': "$Success@10$",
        'SDCG(min_rel=4,max_rel=5)': "SDCG@10$",
    }

    pretty_print_models = {
        'boxe': 'BoxE', 'complex': 'ComplEx', 'hake_both': 'HAKE', 'hole': 'HolE', 'quate': 'QuatE', 'rotate': 'RotatE',
        'rotpro': 'RotPro', 'toruse_eL2': 'TorusE', 'transe': 'TransE', 'tucker': 'TuckER'
    }

    at_k = {
        'Success(rel=5)': 10,
        'SDCG(min_rel=4,max_rel=5)': 10,
    }

    # TODO Uncomment this to get the file the first time.
    # global_models = set()
    # data_by_metric = {}
    #
    # non_singleton_queries = json.load(open(os.path.join(folder_input, 'non_singleton_queries.json')))
    #
    # for metric_name in pretty_print_metrics:
    #     print('Parsing file', metric_name, '--', datetime.now())
    #
    #     filename = os.path.join(folder_input, f'Metrics_{metric_name}.csv')
    #
    #     if os.path.isfile(filename):
    #         df = pd.read_csv(filename, names=['Dataset', 'QueryId', 'EvalType', 'Model', 'Metric', 'k', 'Value'])
    #         df['Positives'] = 1
    #         global_models = global_models.union(set(df['Model'].to_numpy()))
    #
    #         # Add number of positives.
    #         non_singleton_query_ids = {}
    #         for dataset in non_singleton_queries:
    #             non_singleton_query_ids[dataset] = []
    #             positives_histogram = {}
    #
    #             for query_dict in non_singleton_queries[dataset]:
    #                 query_id, pos_count = query_dict['q'], query_dict['count']
    #
    #                 if pos_count not in positives_histogram:
    #                     positives_histogram[pos_count] = []
    #
    #                 positives_histogram[pos_count].append(query_id)
    #                 non_singleton_query_ids[dataset].append(query_id)
    #
    #             for pos_count in positives_histogram:
    #                 df.loc[
    #                     (df['Dataset'] == dataset) &
    #                     (df['QueryId'].isin(positives_histogram[pos_count])), 'Positives'] = pos_count
    #
    #         data_by_metric[metric_name] = df
    #
    # data_to_plot = []
    # for dataset in datasets:
    #     print('Processing:', dataset, '--', datetime.now())
    #
    #     for metric_name in pretty_print_metrics:
    #         print('\tProcessing:', metric_name, '--', datetime.now())
    #
    #         for eval_type in eval_types:
    #             df = data_by_metric[metric_name]
    #             micro_macro, positives_to_include = eval_type.split('_')[0], eval_type.split('_')[1]
    #
    #             for model in global_models:
    #                 filtered_df = df[
    #                     (df['Dataset'] == 'FB15K237' if dataset == 'FB15K-237' else dataset) &
    #                     (df['EvalType'] == micro_macro) & (df['Model'] == model) &
    #                     (df['Metric'].str.startswith(metric_name)) &
    #                     (df['Positives'] >= (2 if positives_to_include == 'non-singleton' else 1)) &
    #                     (df['Positives'] <= (1 if positives_to_include == 'singleton' else 1000000000))
    #                 ]
    #
    #                 if metric_name in at_k:
    #                     filtered_df = filtered_df[filtered_df['k'] == at_k[metric_name]]
    #
    #                 measurements = filtered_df['Value'].to_numpy()
    #
    #                 mean = np.average(measurements)
    #                 std = np.std(measurements)
    #
    #                 data_to_plot.append(
    #                     {'Dataset': dataset,
    #                      'Metric': metric_name, 'EvalType': eval_type,
    #                      'Model': pretty_print_models[model], 'Mean': mean, 'Std': std, 'CV': std/mean})
    #
    # df = pd.DataFrame(data_to_plot)
    # df.to_csv(os.path.join(folder_input, 'micro_macro_data.csv'), index=False)

    df = pd.read_csv(os.path.join(folder_input, 'micro_macro_data.csv'))

    for metric_name in pretty_print_metrics:
        all_mean_dataframes = {}
        for eval_type in eval_types:
            if eval_type == 'micro_all':
                fig_size = (5.50, 3.25)
            else:
                fig_size = (4.25, 3.25)

            plt.figure(figsize=fig_size)
            plt.rcParams.update({'font.size': 11})

            mean_df = df[(df['EvalType'] == eval_type) & (df['Metric'] == metric_name)].pivot_table(
                index='Dataset',
                columns='Model',
                values='Mean',
            )

            # Re-index to preserve desired order.
            mean_df = mean_df.reindex(datasets)

            all_mean_dataframes[eval_type] = mean_df

            # Coefficient of variation.
            cv_df = df[(df['EvalType'] == eval_type) & (df['Metric'] == metric_name)].pivot_table(
                index='Dataset',
                columns='Model',
                values='CV',
            )

            if eval_type == 'macro_all':
                mean_df = mean_df - all_mean_dataframes['micro_all']
            if eval_type == 'macro_non-singleton':
                mean_df = mean_df - all_mean_dataframes['macro_all']

            # annot_df = mean_df.copy().astype(str)
            #
            # for i in range(mean_df.shape[0]):
            #     for j in range(mean_df.shape[1]):
            #         m = mean_df.iloc[i, j]
            #         cv = cv_df.iloc[i, j]
            #
            #         annot_df.iloc[i, j] = m

            if eval_type == 'micro_all':
                ax = sns.heatmap(mean_df, annot=True, linewidth=.5, vmin=0, vmax=.9, fmt='.3f', annot_kws={"size": 7},
                            cbar_kws={'shrink': 0.75, 'pad': 0.01})
            else:
                ax = sns.heatmap(mean_df, annot=True, linewidth=.5, vmin=-.30, vmax=.30, fmt='.3f', annot_kws={"size": 7},
                            cmap='vlag', yticklabels=False, cbar_kws={'shrink': 0.75, 'pad': 0.01})

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=7)
            ax.tick_params(axis='x', rotation=65)

            plt.xlabel('')
            plt.ylabel('')

            plt.tight_layout(pad=0.25, h_pad=0.25, w_pad=0.25)

            pdf_path = os.path.join(folder_input, f'{metric_name}_{eval_type}_Heatmap.pdf')
            with PdfPages(pdf_path, keep_empty=False) as pdf:
                pdf.savefig()
                pdf.close()

            plt.close()


if __name__ == "__main__":
    main()