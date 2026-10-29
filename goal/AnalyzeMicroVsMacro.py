import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import DatasetUtils
from DataLoader import DataLoader

from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages


def main(dataset, data_to_plot):
    # TODO Change this folder to include these datasets: https://github.com/nari97/AugmentedKGE/tree/master/Datasets
    main_folder = ''

    # Only for NELL!
    prefix = ''
    if dataset == 3:
        prefix = '_resplit'

    # Find out model name and dataset.
    reshuffled_id = 0

    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print(f'Dataset: {dataset_name}... --', datetime.now())

    dataset_folder = os.path.join(main_folder, 'Datasets', dataset_name)
    train_loader = DataLoader(dataset_folder, "train")
    val_loader = DataLoader(dataset_folder, "valid")
    test_loader = DataLoader(dataset_folder, "test")
    main_loader = DataLoader(dataset_folder, f'{reshuffled_id}{prefix}_test')

    # manager = TripleManager(
    #     main_loader, train_loader, val_loader, test_loader)

    # Analyze 1-1, 1-M, M-1, M-M (similar to TransE).
    relations = set()
    for loader in [train_loader, val_loader, test_loader]:
        relations = relations.union(loader.relations)

    relation_category_dict = {}
    for current_r in relations:
        heads, tails = {}, {}

        for loader in [train_loader, val_loader, test_loader]:
            for h, r, t in loader.get_triples():
                if current_r == r:
                    if t not in heads:
                        heads[t] = 0
                    heads[t] += 1

                    if h not in tails:
                        tails[h] = 0
                    tails[h] += 1

        avg_head_per_tail, avg_tail_per_head = \
            np.mean(np.array(list(heads.values()))), np.mean(np.array(list(tails.values())))

        if avg_head_per_tail < 1.5:
            r_type = '1-'
        else:
            r_type = 'M-'

        if avg_tail_per_head < 1.5:
            r_type += '1'
        else:
            r_type += 'M'

        relation_category_dict[current_r] = r_type

    cardinality_histogram = {}
    for r in relation_category_dict:
        c = relation_category_dict[r]

        if c not in cardinality_histogram:
            cardinality_histogram[c] = 0

        cardinality_histogram[c] += 1

    print('Relation cardinality histogram:', cardinality_histogram)

    if dataset_name == 'FB15K237':
        dataset_name = 'FB15K-237'

    non_singleton_queries = []
    for evaluation_type in ['micro', 'macro']:
        queries = {}

        # For each triple in test.
        for h, r, t in main_loader.get_triples():
            # We will try two queries: (h, r, ?) and (?, r t).
            # If micro, we try all possible questions. If macro, we group them.
            if evaluation_type == 'micro':
                query_ids = [(f'({h},{r},{t})-h', 'head', h), (f'({h},{r},{t})-t', 'tail', t)]

            if evaluation_type == 'macro':
                query_ids = [(f'(?,{r},{t})', 'head', h), (f'({h},{r},?)', 'tail', t)]

            for query_id, direction, positive in query_ids:
                if query_id not in queries:
                    # Get the pre-computed weight. Each query must have a single one (same predicate).
                    queries[query_id] = {
                        'direction': direction,
                        'positives': [],
                        'h': h,
                        'r': r,
                        't': t,
                    }

                queries[query_id]['positives'].append(positive)

        # current = 0
        # for query_id in sorted(queries):
        #     current += 1
        #
        #     if current % 100 == 0:
        #         print(f'\tDone with {current} queries out of {len(queries)} --', datetime.now())
        #
        #     direction = queries[query_id]['direction']
        #     positives = queries[query_id]['positives']
        #
        #     h, r, t = queries[query_id]['h'], queries[query_id]['r'], queries[query_id]['t']

        print(
            'Evaluation type:', evaluation_type,
            '; head queries:', len([q for q in queries if queries[q]['direction'] == 'head']),
            '; positives per head query:',
                sum([len(queries[q]['positives']) for q in queries if queries[q]['direction'] == 'head']),
            '; tail queries:', len([q for q in queries if queries[q]['direction'] == 'tail']),
            '; positives per tail query:',
                sum([len(queries[q]['positives']) for q in queries if queries[q]['direction'] == 'tail']),
        )

        for query_id in queries:
            if len(queries[query_id]['positives']) > 1:
                non_singleton_queries.append({'q': query_id, 'count': len(queries[query_id]['positives'])})

            data_to_plot['query_id'].append(query_id)
            data_to_plot['evaluation_type'].append(evaluation_type)
            data_to_plot['direction'].append(queries[query_id]['direction'])
            data_to_plot['dataset'].append(dataset_name)
            data_to_plot['positives'].append(len(queries[query_id]['positives']))
            data_to_plot['cardinality'].append(relation_category_dict[queries[query_id]['r']])

    print('Non-singleton queries:', non_singleton_queries)

    print('Done! --', datetime.now())


if __name__ == "__main__":
    # TODO This is where you want your output files.
    output_dir = ''

    data_to_plot = {
        'query_id': [],
        'evaluation_type': [],
        'direction': [],
        'dataset': [],
        'positives': [],
        'cardinality': [],
    }

    main(6, data_to_plot)
    main(3, data_to_plot)
    main(5, data_to_plot)
    main(2, data_to_plot)

    main(0, data_to_plot)
    main(7, data_to_plot)
    main(8, data_to_plot)
    main(9, data_to_plot)

    # main(1, data_to_plot)

    # You can do this once and then reutilize the file.
    dataframe = pd.DataFrame(data_to_plot)

    conditions = [
        (dataframe['positives'] > 1) & (dataframe['evaluation_type'] == 'macro'),
        (dataframe['positives'] == 1) & (dataframe['evaluation_type'] == 'macro'),
        (dataframe['positives'] == 1) & (dataframe['evaluation_type'] == 'micro')
    ]
    reversed(conditions)
    choices = ['Macro non-singleton', 'Macro singleton', 'Micro singleton']
    reversed(choices)
    dataframe['pos_bucket'] = pd.Categorical(
        np.select(conditions, choices, default='Unknown'), categories=choices, ordered=True)

    dataframe['direction'] = dataframe['direction'].replace({
        "head": "Subject",
        "tail": "Object"
    })

    dataframe.to_csv(os.path.join(output_dir, 'micro_macro_queries_data.csv'), index=False)

    # Start here if you already have your file.
    dataframe = pd.read_csv(os.path.join(output_dir, 'micro_macro_queries_data.csv'))

    conditions = [
        (dataframe['positives'] > 1) & (dataframe['evaluation_type'] == 'macro'),
        (dataframe['positives'] == 1) & (dataframe['evaluation_type'] == 'macro'),
        (dataframe['positives'] == 1) & (dataframe['evaluation_type'] == 'micro')
    ]
    reversed(conditions)
    choices = ['Non-singleton', 'Singleton', 'Micro']
    reversed(choices)
    dataframe['pos_bucket'] = pd.Categorical(
        np.select(conditions, choices, default='Unknown'), categories=choices, ordered=True)

    plt.rcParams.update({'font.size': 11})

    direction_order = ['Subject', 'Object']
    bright_palette = sns.color_palette('Accent')[:len(direction_order)]
    dataset_order = ['FB13', 'FB15K', 'FB15K-237', 'WN18', 'WN18RR',  # 'WN11',
                     'BioKG', 'Hetionet', 'NELL-995', 'YAGO3-10']

    # Let's count how many for each dataset:
    for dataset_name in dataset_order:
        total_singleton_queries = len(
            dataframe[(dataframe['pos_bucket'] == 'Singleton') & (dataframe['dataset'] == dataset_name)])
        total_nonsingleton_queries = len(
            dataframe[(dataframe['pos_bucket'] == 'Non-singleton') & (dataframe['dataset'] == dataset_name)])
        print(
            'Dataset:', dataset_name,
            '; Total singleton:', total_singleton_queries,
            '; Total non-singleton:', total_nonsingleton_queries,
            '; Ratio:', total_nonsingleton_queries/(total_singleton_queries + total_nonsingleton_queries),
        )

    g = sns.displot(
        data=dataframe,
        y='pos_bucket', hue='direction', hue_order=direction_order,
        col='dataset', col_order=dataset_order, kind='hist',
        palette=bright_palette,
        height=3.25, aspect=.5, legend=True,
        multiple='dodge', shrink=.75
    )

    g.set(xscale='log')

    g.set_axis_labels('#Queries', '')
    g.set_titles('{col_name}')

    for ax in g.axes.flat:
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.25)

    sns.move_legend(g,  # bbox_to_anchor=(0, 1.05),
                    loc='upper left', title=None, fontsize='small')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5,)

    pdf_path = os.path.join(output_dir, f'MacroPosPerQuery.pdf')
    with PdfPages(pdf_path, keep_empty=False) as pdf:
        pdf.savefig()
        pdf.close()

    plt.close()

    fig_size = (6.5, 4.5)
    plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size': 11})

    eval_order = ['micro', 'macro']
    bright_palette = sns.color_palette('pastel')[:len(eval_order)]

    g = sns.displot(
        data=dataframe, y='cardinality', hue='evaluation_type', hue_order=eval_order,
        col='dataset', col_order=dataset_order, kind='hist',
        palette=bright_palette,
        height=3, aspect=.85, legend=True,
        multiple='dodge', shrink=.8
    )

    g.set(xscale='log')

    g.set_axis_labels('Number of queries', '')
    g.set_titles('{col_name}')

    for ax in g.axes.flat:
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.25)

    sns.move_legend(g, # bbox_to_anchor=(0, 1.05),
                    loc='upper left', title=None, fontsize='small')

    plt.tight_layout()

    pdf_path = os.path.join(output_dir, f'MicroMacroPerPredCardinality.pdf')
    with PdfPages(pdf_path, keep_empty=False) as pdf:
        pdf.savefig()
        pdf.close()

    plt.close()
    #
    # In FB13, these are the values:
    # FB13 top:        evaluation_type direction dataset  positives cardinality           pos_bucket
    # 149812           macro      head    FB13       4931         M-1  Macro non-singleton
    # 149810           macro      head    FB13       2290         M-1  Macro non-singleton
    # 149831           macro      head    FB13        865         M-1  Macro non-singleton
    # 149821           macro      head    FB13        714         M-1  Macro non-singleton
    # 149829           macro      head    FB13        630         M-1  Macro non-singleton
    # 149819           macro      head    FB13        533         M-1  Macro non-singleton
    #
    # Below these, all are under 500. Manually changing those to reduce the plot.
    for top_positives in [4931, 2290, 865, 714, 630, 533]:
        dataframe.loc[(dataframe['dataset'] == 'FB13') & (dataframe['positives'] == top_positives), 'positives'] = 500

    # print('FB13 top:', dataframe[
    #     (dataframe['positives'] > 1) & (dataframe['evaluation_type'] == 'macro') & (dataframe['dataset'] == 'FB13')].
    #       nlargest(25, 'positives').to_string())

    # Trying to get examples from YAGO...
    # print('YAGO top:', dataframe[
    #     (dataframe['positives'] > 1) &
    #     (dataframe['evaluation_type'] == 'macro') &
    #     (dataframe['dataset'] == 'YAGO3-10') &
    #     (dataframe['direction'] == 'tail')
    #     ].nlargest(25, 'positives').to_string())

    # fig_size = (6.5, 4.5)
    # plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size': 11})

    g = sns.displot(
        data=dataframe[(dataframe['positives'] > 1) & (dataframe['evaluation_type'] == 'macro')],
        x='positives', col='dataset', col_order=dataset_order, kind='hist',
        palette=bright_palette, height=3.25, aspect=0.50, shrink=.75,
    )

    g.set(yscale='log', xscale='log')

    g.set_axis_labels('#Answers', '#Queries')
    g.set_titles('{col_name}')

    for ax in g.axes.flat:
        ax.grid(True, axis='both', color='gray', linestyle='--', linewidth=0.25)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5,)

    pdf_path = os.path.join(output_dir, f'NonSingletonMacroHistogram.pdf')
    with PdfPages(pdf_path, keep_empty=False) as pdf:
        pdf.savefig()
        pdf.close()

    plt.close()