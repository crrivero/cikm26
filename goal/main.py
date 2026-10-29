import json
import logging
import os
import sys
from datetime import datetime

import ir_measures
import numpy as np
from ir_measures.providers import PytrecEvalProvider, CwlEvalProvider, JudgedProvider, PyNdEvalProvider
from ir_measures.providers.cwl_eval import IrmQrelHandler

import DatasetUtils
from DataLoader import DataLoader
from TripleManager import TripleManager


def main():
    start_query_id, end_query_id = None, None

    # TODO Model files are expected here. Check the ranks.tsv files.
    main_folder = sys.argv[1]
    array_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        start_query_id, end_query_id = int(sys.argv[3]), int(sys.argv[4])

    batch_size = 5

    # 'regular': pos=5, dr=4..1 and rest left unjudged
    # 'regular_n_neg': pos=5, dr=4..1 and rest=1
    # 'pos_only': pos=5 and rest left unjudged
    # 'pos_n_neg': pos=5 and rest=1
    # 'dr_only': pos=5, dr=4 and rest unjudged
    # 'dr_n_neg': pos=5, dr=4 and rest=1
    relevance_type = 'regular'

    # These are the relevance values to consider.
    # This indicates whether to consider judged_only or not.
    if 'regular' in relevance_type:
        bin_rel_values, nonbin_rel_values = [3, 4, 5], [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 6)]
        include_judged_only = True

    if relevance_type == 'pos_only':
        bin_rel_values, nonbin_rel_values = [5], [(0, 5), (1, 5)]
        include_judged_only = True

    if relevance_type == 'pos_n_neg':
        bin_rel_values, nonbin_rel_values = [5], [(0, 5), (1, 5)]
        include_judged_only = False

    # Define evaluation measures
    base_measures = []

    for rel in bin_rel_values:
        base_measures.append(ir_measures.AP(rel=rel))
        base_measures.append(ir_measures.BPref(rel=rel))
        for p in [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]:
            base_measures.append(ir_measures.RBP(rel=rel, p=p))
        base_measures.append(ir_measures.Rprec(rel=rel))

        base_measures.append(ir_measures.RR(rel=rel))
        base_measures.append(ir_measures.SetP(rel=rel))
        base_measures.append(ir_measures.SetR(rel=rel))

        if include_judged_only:
            base_measures.append(ir_measures.RR(rel=rel, judged_only=True))
            base_measures.append(ir_measures.SetP(rel=rel, judged_only=True))

    base_measures.append(ir_measures.Judged)

    for k in [
        # pyndeval doesn't allow more than 20!
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        # 50, 75, 100, 250, 500, 750, 1000
    ]:
        for rel in bin_rel_values:
            base_measures.append(ir_measures.P(rel=rel, judged_only=False)@k)
            base_measures.append(ir_measures.R(rel=rel, judged_only=False)@k)
            base_measures.append(ir_measures.Success(rel=rel, judged_only=False)@k)

            # Equivalent to nDCG.
            # base_measures.append(ir_measures.alpha_nDCG(rel=min_rel, judged_only=False, alpha=.0001)@k)

            if include_judged_only:
                base_measures.append(ir_measures.P(rel=rel, judged_only=True)@k)
                base_measures.append(ir_measures.R(rel=rel, judged_only=True)@k)
                base_measures.append(ir_measures.Success(rel=rel, judged_only=True)@k)

                # base_measures.append(ir_measures.alpha_nDCG(rel=min_rel, judged_only=True, alpha=.0001)@k)

        for min_rel, max_rel in nonbin_rel_values:
            base_measures.append(ir_measures.BPM(min_rel=min_rel, max_rel=max_rel)@k)
            base_measures.append(ir_measures.SDCG(min_rel=min_rel, max_rel=max_rel)@k)

        base_measures.append(ir_measures.nDCG(judged_only=False)@k)
        if include_judged_only:
            base_measures.append(ir_measures.nDCG(judged_only=True)@k)

    pytrec_provider = PytrecEvalProvider()
    cwl_provider = CwlEvalProvider()
    judged_provider = JudgedProvider()
    # pyndeval_provider = PyNdEvalProvider()

    pytrec_measures, cwl_measures, judged_measures, pyndeval_measures = set(), set(), set(), set()
    for measure in base_measures:
        if measure is ir_measures.Judged:
            judged_measures.add(measure)
        elif pytrec_provider.supports(measure):
            pytrec_measures.add(measure)
        elif cwl_provider.supports(measure):
            cwl_measures.add(measure)
        # elif pyndeval_provider.supports(measure):
        #     pyndeval_measures.add(measure)
        else:
            print(f'Measure {measure} not supported!')

    # Find out model name and dataset.
    def find_setup():
        current = 0
        for reshuffled_id in [0]: #range(0, 26):
            if 'regular' in relevance_type:
                method = 'dice'
            else:
                method = None

            for dataset in range(0, 10):
                for evaluation_type in ['micro', 'macro']:
                    if current == array_id:
                        return reshuffled_id, method, dataset, evaluation_type
                    else:
                        current += 1

        return None

    reshuffled_id, method, dataset, evaluation_type = find_setup()

    # Only for NELL!
    prefix = ''
    if dataset == 3:
        prefix = '_resplit'

    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print(f'Processing {dataset_name} and reshuffle {reshuffled_id} '
          f'using {method} with {evaluation_type} setting')

    print('Metadata:',
          json.dumps(
              {'dataset': dataset_name, 'compatible_method': method,
               'relevance_type': relevance_type, 'evaluation_type': evaluation_type}))

    model_names = ['boxe', 'complex', 'hake_both', 'hole', 'quate', 'rotate',
                   'rotpro', 'toruse_eL2', 'transe', 'tucker']
    measure_names = [str(m) for m in base_measures]

    model_to_index = {idx: m for m, idx in enumerate(model_names)}
    measure_to_index = {idx: m for m, idx in enumerate(measure_names)}

    print('Metadata:', json.dumps({'metrics': measure_names, 'models': model_names}))

    print('Loading data using data loaders... --', datetime.now())

    dataset_folder = os.path.join(main_folder, 'Datasets', dataset_name)
    train_loader = DataLoader(dataset_folder, "train")
    val_loader = DataLoader(dataset_folder, "valid")
    test_loader = DataLoader(dataset_folder, "test")

    # Let's collect the weights.
    collected_weights = []

    main_loader = DataLoader(dataset_folder, f'{reshuffled_id}{prefix}_test')

    print('Loading data using data triple manager... --', datetime.now())

    manager = TripleManager(
        main_loader, train_loader, val_loader, test_loader,
        similarity_method=method)

    print('Loading ranks... --', datetime.now())

    run_dicts = {}

    for model_name in model_names:
        model_ranks_file = os.path.join(
            main_folder, 'Model', str(dataset),
            f'{model_name}{prefix}__{reshuffled_id}_ranks.tsv')

        if os.path.exists(model_ranks_file):
            run_dict = {}

            with open(model_ranks_file, 'r', encoding='utf-8') as file:
                for line in file:
                    json_object = json.loads(line.strip())
                    query_id = json_object['q']

                    if query_id not in run_dict:
                        run_dict[query_id] = {}

                    # Take only the top!
                    entities_of_interest = np.array(json_object['te'])
                    scores_of_interest = np.array(json_object['ts'])

                    # entities_of_interest = np.array(json_object['be'])
                    # scores_of_interest = np.array(json_object['bs'])

                    run_dict[query_id]['entities_of_interest'] = entities_of_interest
                    run_dict[query_id]['scores_of_interest'] = scores_of_interest

            run_dicts[model_name] = run_dict

    print('Ranks ready! --', datetime.now())

    print('Calculating IR measures... --', datetime.now())

    triples = manager.get_triples()

    total_qrel_gain_minmax, total_qrel_rels_minmax = {}, {}
    batched_qrels_dict, batched_qrels_dict_minmax, batched_runs = {}, {}, {}

    # We are assuming relevance is always between 1 and 5.
    batched_qrels_dict_binary = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    total_qrel_gain_binary = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    total_qrel_rels_binary = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

    def process_batch():
        if len(batched_qrels_dict) > 0:
            # No warnings, please!
            logger = logging.getLogger('ir_measures.cwl_eval')
            logger.setLevel('ERROR')

            pytrec_evaluator = pytrec_provider.evaluator(pytrec_measures, batched_qrels_dict)
            judged_evaluator = judged_provider.evaluator(judged_measures, batched_qrels_dict)
            # pyndeval_evaluator = pyndeval_provider.evaluator(pyndeval_measures, batched_qrels_dict)
            # Don't panic! The batched_qrels_dict is applied below, that is why the list provided is empty.
            cwl_evaluator = cwl_provider.evaluator(cwl_measures, [])

            for inv_key in cwl_evaluator.qrhs:
                binary_cutoff = inv_key[0]
                min_value, max_value = inv_key[1], inv_key[2]

                cwl_evaluator.min_rel = min_value
                cwl_evaluator._min_observed_rel = min_value

                cwl_evaluator.max_rel = max_value
                cwl_evaluator._max_observed_rel = max_value

                cwl_evaluator.qrhs[inv_key] = IrmQrelHandler(*inv_key)

                if binary_cutoff is None:
                    cwl_evaluator.qrhs[inv_key].data.update(batched_qrels_dict_minmax)
                    cwl_evaluator.qrel_qids = set(batched_qrels_dict_minmax.keys())
                else:
                    # This binary gain must be chosen based on the binary_cutoff!
                    cwl_evaluator.qrhs[inv_key].data.update(batched_qrels_dict_binary[binary_cutoff])
                    cwl_evaluator.qrel_qids = set(batched_qrels_dict_binary[binary_cutoff].keys())

            # Alternative iter_calc.
            def _alt_iter_calc(run):
                from cwl.cwl_eval import RankingMaker
                from cwl.cwl_eval import Ranking

                for qid in run.keys():
                    # Create the ranking makers.
                    ranking_makers = {}
                    for inv_key in cwl_evaluator.invocations.keys():
                        ranking_makers[inv_key] = RankingMaker(qid, cwl_evaluator.qrhs[inv_key], min_gain=0.,
                                                               max_gain=1., cost_dict=None, max_cost=1.0,
                                                               min_cost=1.0, max_n=1000)

                    sorted_items = sorted(
                        [{'doc': doc_id, 'score': run[qid][doc_id]}
                         for doc_id in run[qid].keys()], key=lambda x: x['score'], reverse=True)

                    for item in sorted_items:
                        for maker in ranking_makers.values():
                            maker.add(item['doc'], '_')

                    rankings = {}

                    for inv_key, maker in ranking_makers.items():
                        ranking = Ranking(maker.topic_id, maker._gains, maker._costs, maker.max_gain, maker.min_gain,
                                          maker.max_cost, maker.min_cost, maker.max_n)

                        if inv_key[0] is not None:
                            ranking.total_qrel_gain = total_qrel_gain_binary[inv_key[0]][qid]
                            ranking.total_qrel_rels = total_qrel_rels_binary[inv_key[0]][qid]
                        else:
                            ranking.total_qrel_gain = total_qrel_gain_minmax[qid]
                            ranking.total_qrel_rels = total_qrel_rels_minmax[qid]

                        rankings[inv_key] = ranking

                    for inv_key, meas in cwl_evaluator.invocations.items():
                        for measure in meas:
                            cwl_measure = cwl_evaluator._irm_convert_to_measure(measure)
                            value = cwl_measure.measure(rankings[inv_key])
                            yield ir_measures.providers.cwl_eval.CwlMetric(
                                query_id=qid, measure=measure, value=value,
                                expected_total_utility=cwl_measure.expected_total_utility,
                                expected_cost=cwl_measure.expected_cost,
                                expected_total_cost=cwl_measure.expected_total_cost,
                                expected_items=cwl_measure.expected_items)

            cwl_evaluator._iter_calc = _alt_iter_calc

            # Collect metrics and print.
            collected_measures = {}

            for model_name in batched_runs:
                for eval in [
                    pytrec_evaluator, cwl_evaluator, judged_evaluator, # pyndeval_evaluator,
                ]:
                    for metric in eval.iter_calc(batched_runs[model_name]):
                        query_id, metric_value = metric.query_id, float(metric.value)

                        if query_id not in collected_measures:
                            collected_measures[query_id] = {}

                        model_idx = int(model_to_index[model_name])
                        if model_idx not in collected_measures[query_id]:
                            collected_measures[query_id][model_idx] = {}

                        # Only print if greater than zero.
                        if metric_value > 0:
                            metric_idx = int(measure_to_index[str(metric.measure)])
                            collected_measures[query_id][model_idx][metric_idx] = metric_value

            for query_id in collected_measures:
                print('D:', json.dumps({query_id: collected_measures[query_id]}))

            batched_qrels_dict.clear()
            batched_qrels_dict_minmax.clear()
            batched_runs.clear()

            total_qrel_gain_minmax.clear()
            total_qrel_rels_minmax.clear()

            for cutoff in batched_qrels_dict_binary.keys():
                batched_qrels_dict_binary[cutoff].clear()
                total_qrel_gain_binary[cutoff].clear()
                total_qrel_rels_binary[cutoff].clear()

    queries, queries_to_merge = {}, {}

    # For each triple in test.
    for h, r, t in triples:
        # We will try two queries: (h, r, ?) and (?, r t).
        # If micro, we try all possible questions. If macro, we group them.
        if evaluation_type == 'micro':
            query_ids = [(f'({h},{r},{t})-h', 'head', h), (f'({h},{r},{t})-t', 'tail', t)]

        if evaluation_type == 'macro':
            query_ids = [(f'(?,{r},{t})', 'head', h), (f'({h},{r},?)', 'tail', t)]

            # Indicate here the queries we must merge! Then, update run_dicts accordingly.
            # I know, I know, we could do this a little bit better. Spaghetti code!
            if f'(?,{r},{t})' not in queries_to_merge:
                queries_to_merge[f'(?,{r},{t})'] = [f'({h},{r},{t})-h']
            else:
                queries_to_merge[f'(?,{r},{t})'].append(f'({h},{r},{t})-h')

            if f'({h},{r},?)' not in queries_to_merge:
                queries_to_merge[f'({h},{r},?)'] = [f'({h},{r},{t})-t']
            else:
                queries_to_merge[f'({h},{r},?)'].append(f'({h},{r},{t})-t')

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

    if len(queries_to_merge) > 0:
        # Merge run_dicts!
        for m in run_dicts:
            new_run_dicts = {}

            for qid in queries_to_merge:
                new_run_dicts[qid] = {'entities_of_interest': np.array([]), 'scores_of_interest': np.array([])}

                for other_qid in queries_to_merge[qid]:
                    new_run_dicts[qid]['entities_of_interest'] = np.append(
                        new_run_dicts[qid]['entities_of_interest'], run_dicts[m][other_qid]['entities_of_interest'])
                    new_run_dicts[qid]['scores_of_interest'] = np.append(
                        new_run_dicts[qid]['scores_of_interest'], run_dicts[m][other_qid]['scores_of_interest'])

            run_dicts[m] = new_run_dicts

    # For each triple in test.
    sorted_query_ids = sorted(queries)
    for idx, query_id in enumerate(sorted_query_ids):
        if start_query_id is not None and (idx < start_query_id or idx >= end_query_id):
            continue

        if idx % 100 == 0:
            print(f'\tDone with {idx} queries out of {len(queries)} --', datetime.now())

        direction = queries[query_id]['direction']
        positives = queries[query_id]['positives']

        h, r, t = queries[query_id]['h'], queries[query_id]['r'], queries[query_id]['t']

        # Get the negatives.
        entities_by_relevance = {}

        if 'regular' in relevance_type:
            selected_entities = manager.selected_entities_by_rd[r][direction]['entities']
            relevance = manager.selected_entities_by_rd[r][direction][relevance_type]

            for i, e in enumerate(selected_entities):
                entities_by_relevance[e] = relevance[i]

            # Add positive(s) assuming relevance is 5. If there are repeated entities, this makes sure 5 prevails.
            for p in positives:
                entities_by_relevance[p] = 5

            for e in manager.get_corrupted(h, r, t, corruption_type=direction):
                if e not in entities_by_relevance:
                    entities_by_relevance[e] = 1 if relevance_type == 'regular_n_neg' else 0

        elif 'pos_' in relevance_type:
            # pos=5 and rest left unjudged
            for p in positives:
                entities_by_relevance[p] = 5

            # pos=5 and rest=1
            for e in manager.get_corrupted(h, r, t, corruption_type=direction):
                if e not in entities_by_relevance:
                    entities_by_relevance[e] = 1 if relevance_type == 'pos_n_neg' else 0

        relevance_values = np.array(list(entities_by_relevance.values()), dtype=int)

        # Sanity checks. There should be exactly one for micro and one or more for macro.
        if evaluation_type == 'micro' and (relevance_values == 5).sum(axis=0) != 1:
            print('Relevance 5 is incorrect:', (relevance_values == 5).sum(axis=0))

        if evaluation_type == 'macro' and (relevance_values == 5).sum(axis=0) == 0:
            print('Relevance 5 is incorrect:', (relevance_values == 5).sum(axis=0))

        GetLinePlots.py# Get the qrels and gains.
        batched_qrels_dict[query_id] = {}
        batched_qrels_dict_minmax[query_id] = {}

        for cutoff in batched_qrels_dict_binary.keys():
            batched_qrels_dict_binary[cutoff][query_id] = {}
            total_qrel_gain_binary[cutoff][query_id] = {}

        total_qrel_rels_minmax[query_id] = 0
        total_qrel_gain_minmax[query_id] = 0

        for current_relevance in range(1, 6):
            total_qrel_gain_binary[current_relevance][query_id] = 0
            total_qrel_rels_binary[current_relevance][query_id] = 0

        for e in entities_by_relevance:
            e_rel = int(entities_by_relevance[e])
            e_str = str(e)

            # We are assuming relevance is always between 1 and 5.
            minmax_gain = float((e_rel - 1) / (5 - 1))

            batched_qrels_dict[query_id][e_str] = e_rel
            batched_qrels_dict_minmax[query_id][e_str] = minmax_gain

            for current_relevance in range(1, 6):
                binary_gain = 1 if e_rel >= current_relevance else 0
                batched_qrels_dict_binary[current_relevance][query_id][e_str] = binary_gain

                if binary_gain > 0.0:
                    total_qrel_gain_binary[current_relevance][query_id] += binary_gain
                    total_qrel_rels_binary[current_relevance][query_id] += 1

            if minmax_gain > 0.0:
                total_qrel_gain_minmax[query_id] += minmax_gain
                total_qrel_rels_minmax[query_id] += 1

        # Collect metrics.
        for model_name in run_dicts:
            if model_name not in batched_runs:
                batched_runs[model_name] = {}

            batched_runs[model_name][query_id] = {}
            for idx, e in enumerate(run_dicts[model_name][query_id]['entities_of_interest']):
                batched_runs[model_name][query_id][str(int(e))] = run_dicts[model_name][query_id]['scores_of_interest'][idx]

        # Collect the weight.
        collected_weights.append(queries[query_id]['weight'])

        if len(batched_qrels_dict) >= batch_size:
            process_batch()

    # Final call!
    process_batch()

    print('Weights:', collected_weights)

    print('Done!... --', datetime.now())


if __name__ == "__main__":
    main()
