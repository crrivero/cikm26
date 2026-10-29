import json
import os.path

import numpy as np
import pandas as pd
from datetime import datetime


def main():
    # TODO Your results after running ProcessMeasurements.py.
    folder_input = ''
    include_singletons, include_non_singletons = True, True
    positive_buckets = \
        [(1, 1), (2, 5), (6, 10), (11, 25), (26, 10000000)]
        #[(1, 10000000)] # We want all (no buckets)

    eval_types = \
        ['macro']  # Only macro
        #['micro', 'macro'] # We want both.
    datasets = [
        'FB13', 'FB15K',
        'FB15K237', 'WN18', 'WN18RR', #'WN11',
        'BioKG', 'Hetionet', 'NELL-995', 'YAGO3-10'
    ]

    pretty_print_metrics = {
        'RR(rel=5)': "$RR$",
        'Bpref(rel=5)': "$Bpref$",
        'Rprec(rel=5)': "$Rprec$",
        'AP(rel=5)': "$AP$",
    }

    at_k = {
        'nDCG(judged_only=True)': 10,
        'SDCG(min_rel=1,max_rel=5)': 10,
        'P(rel=4,judged_only=True)': 10,
        'P(judged_only=True)': 10,
        'R(rel=4,judged_only=True)': 10,
        'R(judged_only=True)': 10,
    }

    pretty_print_models = {
        'boxe': 'Bo', 'complex': 'Co', 'hake_both': 'HA', 'hole': 'Ho', 'quate': 'Qu', 'rotate': 'Rt',
        'rotpro': 'RP', 'toruse_eL2': 'To', 'transe': 'Tr', 'tucker': 'Tu'
    }

    non_singleton_queries = json.load(open(os.path.join(folder_input, 'non_singleton_queries.json')))

    global_models = set()
    data_by_metric = {}

    for metric_name in pretty_print_metrics:
        print('Parsing file', metric_name, '--', datetime.now())

        filename = os.path.join(folder_input, f'Metrics_{metric_name}.csv')

        if os.path.isfile(filename):
            df = pd.read_csv(filename, names=['Dataset', 'QueryId', 'EvalType', 'Model', 'Metric', 'k', 'Value'])
            df['Positives'] = 1

            # Add number of positives.
            non_singleton_query_ids = {}
            for dataset in non_singleton_queries:
                non_singleton_query_ids[dataset] = []
                positives_histogram = {}

                for query_dict in non_singleton_queries[dataset]:
                    query_id, pos_count = query_dict['q'], query_dict['count']

                    if pos_count not in positives_histogram:
                        positives_histogram[pos_count] = []

                    positives_histogram[pos_count].append(query_id)
                    non_singleton_query_ids[dataset].append(query_id)

                for pos_count in positives_histogram:
                    df.loc[
                        (df['Dataset'] == dataset) &
                        (df['QueryId'].isin(positives_histogram[pos_count])), 'Positives'] = pos_count

            if not include_singletons:
                filtered = []
                for dataset in non_singleton_query_ids:
                    filtered.append(df[(df['Dataset'] == dataset) & (df['QueryId'].isin(non_singleton_query_ids[dataset]))])
                df = pd.concat(filtered, ignore_index=True)

            if not include_non_singletons:
                filtered = []
                for dataset in non_singleton_query_ids:
                    filtered.append(df[(df['Dataset'] == dataset) & (~df['QueryId'].isin(non_singleton_query_ids[dataset]))])
                df = pd.concat(filtered, ignore_index=True)

            data_by_metric[metric_name] = df
            global_models = global_models.union(set(df['Model'].to_numpy()))



    data_by_dataset_metric = {}
    for dataset in datasets:
        print('Processing:', dataset, '--', datetime.now())

        data_by_dataset_metric[dataset] = {}

        for metric_name in pretty_print_metrics:
            print('\tProcessing:', metric_name, '--', datetime.now())

            if metric_name not in data_by_dataset_metric[dataset]:
                data_by_dataset_metric[dataset][metric_name] = {}

            for eval_type in eval_types:
                if eval_type not in data_by_dataset_metric[dataset][metric_name]:
                    data_by_dataset_metric[dataset][metric_name][eval_type] = {}

                df = data_by_metric[metric_name]

                for min_value, max_value in positive_buckets:
                    df_by_bucket = df[(df['Positives'] >= min_value) & (df['Positives'] <= max_value)]

                    if (min_value, max_value) not in data_by_dataset_metric[dataset][metric_name][eval_type]:
                        data_by_dataset_metric[dataset][metric_name][eval_type][(min_value, max_value)] = {}

                    for model in global_models:
                        filtered_df = df_by_bucket[
                            (df_by_bucket['Dataset'] == dataset) & (df_by_bucket['EvalType'] == eval_type) &
                            (df_by_bucket['Model'] == model) & (df_by_bucket['Metric'].str.startswith(metric_name))]

                        if metric_name in at_k:
                            filtered_df = filtered_df[filtered_df['k'] == at_k[metric_name]]
                        measurements = filtered_df['Value'].to_numpy()

                        mean = np.average(measurements)
                        std = np.std(measurements)

                        data_by_dataset_metric[dataset][metric_name][eval_type][(min_value, max_value)][model] = \
                            (mean, std)

    print('Printing table', '--', datetime.now())

    metrics_to_print = list(pretty_print_metrics.keys())
    for dataset in datasets:
        print('Dataset:', dataset)

        for m_idx, metric_name in enumerate(metrics_to_print):
            print('\tMetric:', metric_name)

            for min_value, max_value in positive_buckets:
                print('\t\tBucket:', (min_value, max_value))

                for model in global_models:
                    print('\t\t\tModel:', model)

                    for eval_type in eval_types:
                        mean, std = data_by_dataset_metric[dataset][metric_name][eval_type][(min_value, max_value)][model]

                        print('\t\t\t\tType:', eval_type, ', Mean:', mean, 'Std:', std)


if __name__ == "__main__":
    main()