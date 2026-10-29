import os.path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json


def main():
    # TODO The files you want to process.
    folder_input = ''
    # TODO The folder where you want your results.
    output_dir = 'C:/Users/crrvcs/Downloads/CIKM26/'
    eval_type, use_nonsingletons_only = 'macro', False
    rel_str = '5' # This is the relevance we will use.

    dataset_order = ['FB13', 'FB15K', 'FB15K-237', 'WN18', 'WN18RR', 'BioKG', 'Hetionet', 'NELL-995', 'YAGO3-10']

    pretty_print_models = {
        'boxe': 'BoxE', 'complex': 'ComplEx', 'hake_both': 'HAKE', 'hole': 'HolE', 'quate': 'QuatE', 'rotate': 'RotatE',
        'rotpro': 'RotPro', 'toruse_eL2': 'TorusE', 'transe': 'TransE', 'tucker': 'TuckER',
    }

    pretty_print_datasets = {
        'FB13': 'FB13', 'FB15K': 'FB15K', 'FB15K237': 'FB15K-237', 'WN18': 'WN18', 'WN18RR': 'WN18RR',
        'BioKG': 'BioKG', 'Hetionet': 'Hetionet', 'NELL-995': 'NELL-995', 'YAGO3-10': 'YAGO3-10',
    }

    # TODO Run this the first time.
    # non_singleton_queries = json.load(open(os.path.join(folder_input, 'non_singleton_queries.json')))
    #
    # non_singleton_queries["FB15K-237"] = non_singleton_queries["FB15K237"]
    #
    # all_dataframes = []
    # for rbp_p in ['0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5']:
    #     metrics_filename = f'Metrics_RBP(rel={rel_str},p={rbp_p}).csv'
    #     if rbp_p == '0.8':
    #         metrics_filename = f'Metrics_RBP(rel={rel_str}).csv'
    #
    #     filename = os.path.join(folder_input, metrics_filename)
    #
    #     if os.path.isfile(filename):
    #         df = pd.read_csv(filename, names=['Dataset', 'QueryId', 'EvalType', 'Model', 'Metric', 'k', 'Value'])
    #         df = df[df['EvalType'] == eval_type]
    #
    #         df['Model'] = df['Model'].map(pretty_print_models)
    #         df['Dataset'] = df['Dataset'].map(pretty_print_datasets)
    #         df['p'] = float(rbp_p)
    #
    #         df['Positives'] = 1
    #
    #         # Add number of positives.
    #         for dataset in non_singleton_queries:
    #             positives_histogram = {}
    #
    #             for query_dict in non_singleton_queries[dataset]:
    #                 query_id, pos_count = query_dict['q'], query_dict['count']
    #
    #                 if pos_count not in positives_histogram:
    #                     positives_histogram[pos_count] = []
    #
    #                 positives_histogram[pos_count].append(query_id)
    #
    #             for pos_count in positives_histogram:
    #                 df.loc[
    #                     (df['Dataset'] == dataset) &
    #                     (df['QueryId'].isin(positives_histogram[pos_count])), 'Positives'] = pos_count
    #
    #         all_dataframes.append(df)
    #
    # df = pd.concat(all_dataframes)
    #
    # df.to_csv(os.path.join(output_dir, f'{eval_type}_rbp_data.csv'), index=False)

    df = pd.read_csv(os.path.join(output_dir, f'{eval_type}_rbp_data.csv'))

    if use_nonsingletons_only:
        df = df[df['Positives'] > 1]

    plt.rcParams.update({'font.size': 11})

    g = sns.relplot(
        data=df, x='p', col='Dataset', col_order=dataset_order, y='Value', hue='Model', style='Model',
        markers=True, dashes=False, kind='line', palette='pastel', legend=True, linewidth=1.25, markersize=4,
        height=3.25, aspect=0.50, markeredgewidth=0.5,)

    #g.set_axis_labels('Number of positives', 'Number of queries')
    g.set_titles('{col_name}', pad=2.5)

    # plt.xticks(ticks=[i for i in range(1, 26)])
    plt.yticks(ticks=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    plt.ylim(0, .5)
    plt.xlim(1, .5)

    g.tick_params(axis='x', labelsize=7)

    for ax in g.axes.flat:
        ax.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.25)
        ax.tick_params(axis='x', rotation=25)

    g.set(xlabel=r"$\pi$")

    sns.move_legend(
        g, "upper center",
        # "lower center",
        bbox_to_anchor=(.5, 1.03),
        ncol=10, title=None, frameon=False,
    )

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 0.95])

    pdf_path = os.path.join(output_dir, f'RBP_P_{eval_type}_{use_nonsingletons_only}_Plot.pdf')
    with PdfPages(pdf_path, keep_empty=False) as pdf:
        pdf.savefig()
        pdf.close()

    plt.close()


if __name__ == "__main__":
    main()