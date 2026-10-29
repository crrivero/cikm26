import numpy as np
from collections import defaultdict
from functools import cmp_to_key
from CompatibleRelationsGenerator import CompatibleRelationsGenerator


class TripleManager:
    def __init__(self, main_loader, *secondary_loaders, compatible_threshold=0.75, similarity_method="overlap",
                 alpha=0.5, beta=0.5):
        """
        Initializes the TripleManager with a main data loader and optional secondary loaders.

        :param main_loader: The primary DataLoader instance
        :param secondary_loaders: Optional secondary DataLoader instances
        """
        self.main_loader = main_loader
        self.secondary_loaders = secondary_loaders

        # Aggregating structures across all loaders
        self.head_dict = defaultdict(lambda: defaultdict(set))
        self.tail_dict = defaultdict(lambda: defaultdict(set))
        self.domain = defaultdict(set)
        self.range = defaultdict(set)
        self.entities = np.array(list(main_loader.entities))

        # variables for compatibility stuff
        self.threshold = compatible_threshold
        self.similarity_method = similarity_method

        # Unioning all structures
        self._aggregate_structures()

        # Convert sets to NumPy arrays for efficiency
        self._convert_to_numpy()

        # Generate compatible relations using existing dictionaries
        generator = CompatibleRelationsGenerator(self.head_dict, self.tail_dict, self.domain, self.range)
        # generator.generate() # This is for when we want to use the default values and save file
        # This is where it takes the parameters for the compatible relations
        # Parameters like Threshold and Method0
        generator.compute_compatible_relations(threshold=compatible_threshold,
                                               method=similarity_method,
                                               alpha=alpha,
                                               beta=beta)

        self.dom_dom = generator.domDomCompatible
        self.dom_ran = generator.domRanCompatible
        self.ran_dom = generator.ranDomCompatible
        self.ran_ran = generator.ranRanCompatible
        self.all_info_similarities = generator.all_info_similarities

        # Get the pre-made compatible relations dictionaries
        self.compatible_relations = self._build_compatible_relations()

        # Generate relation hardness.
        all_relations = set()
        all_relations = all_relations.union(self.domain.keys())
        all_relations = all_relations.union(self.range.keys())

        all_entities = set()
        for r in all_relations:
            all_entities = all_entities.union(self.domain.get(r, []))
            all_entities = all_entities.union(self.range.get(r, []))

        self.entity_degree, self.entity_degree_by_relation = {}, {}
        for e in all_entities:
            degree = 0

            for r in all_relations:
                if r not in self.entity_degree_by_relation:
                    self.entity_degree_by_relation[r] = {}

                out_degree = len(self.head_dict[r].get(e, []))
                in_degree = len(self.tail_dict[r].get(e, []))

                self.entity_degree_by_relation[r][e] = out_degree + in_degree

                degree += out_degree + in_degree

            self.entity_degree[e] = degree

        # Number of bins to create.
        number_of_bins = 3

        # By relation and direction.
        self.selected_entities_by_rd = {}

        self.total_relevance_by_direction = {'head': {}, 'tail': {}}

        for bin_idx in range(0, number_of_bins):
            self.total_relevance_by_direction['head'][bin_idx] = 0
            self.total_relevance_by_direction['tail'][bin_idx] = 0

        for r in main_loader.relations:
            self.selected_entities_by_rd[r] = {}

            for direction in ['head', 'tail']:
                corrupted_scores = {}
                corrupted_sim_info = {} # You can do {} or None, None is for production.

                if direction == 'head':
                    # This is the maximum similarity.
                    for e in self._get_elements(r, 'domain'):
                        corrupted_scores[e] = 1.0
                        if corrupted_sim_info is not None:
                            corrupted_sim_info[e] = 'domain'

                    for sim_info in self.all_info_similarities:
                        # {'ri': ri, 'ri_type': 'range', 'rj': rj, 'rj_type': 'domain', 'sim': sim}
                        if sim_info['ri'] == r and sim_info['ri_type'] == 'domain' and sim_info['sim'] > 0:
                            for e in self._get_elements(sim_info['rj'], sim_info['rj_type']):
                                if e not in corrupted_scores:
                                    corrupted_scores[e] = 0.0

                                if sim_info['sim'] > corrupted_scores[e]:
                                    corrupted_scores[e] = sim_info['sim']
                                    if corrupted_sim_info is not None:
                                        corrupted_sim_info[e] = sim_info

                        if sim_info['rj'] == r and sim_info['rj_type'] == 'domain' and sim_info['sim'] > 0:
                            for e in self._get_elements(sim_info['ri'], sim_info['ri_type']):
                                if e not in corrupted_scores:
                                    corrupted_scores[e] = 0.0

                                if sim_info['sim'] > corrupted_scores[e]:
                                    corrupted_scores[e] = sim_info['sim']
                                    if corrupted_sim_info is not None:
                                        corrupted_sim_info[e] = sim_info
                elif direction == 'tail':
                    # This is the maximum similarity.
                    for e in self._get_elements(r, 'range'):
                        corrupted_scores[e] = 1.0
                        if corrupted_sim_info is not None:
                            corrupted_sim_info[e] = 'range'

                    for sim_info in self.all_info_similarities:
                        if sim_info['ri'] == r and sim_info['ri_type'] == 'range' and sim_info['sim'] > 0:
                            for e in self._get_elements(sim_info['rj'], sim_info['rj_type']):
                                if e not in corrupted_scores:
                                    corrupted_scores[e] = 0.0

                                if sim_info['sim'] > corrupted_scores[e]:
                                    corrupted_scores[e] = sim_info['sim']
                                    if corrupted_sim_info is not None:
                                        corrupted_sim_info[e] = sim_info

                        if sim_info['rj'] == r and sim_info['rj_type'] == 'range' and sim_info['sim'] > 0:
                            for e in self._get_elements(sim_info['ri'], sim_info['ri_type']):
                                if e not in corrupted_scores:
                                    corrupted_scores[e] = 0.0

                                if sim_info['sim'] > corrupted_scores[e]:
                                    corrupted_scores[e] = sim_info['sim']
                                    if corrupted_sim_info is not None:
                                        corrupted_sim_info[e] = sim_info

                def compare_to_asc(a, b):
                    return (a > b) - (a < b)

                def compare_to_desc(a, b):
                    return (b > a) - (b < a)

                def compare_entities(ei, ej):
                    eiej_cmp = compare_to_desc(corrupted_scores[ei], corrupted_scores[ej])

                    if eiej_cmp == 0:
                        eiej_cmp = compare_to_desc(
                            self.entity_degree_by_relation[r][ei], self.entity_degree_by_relation[r][ej])

                    if eiej_cmp == 0:
                        eiej_cmp = compare_to_desc(
                            self.entity_degree[ei], self.entity_degree[ej])

                    if eiej_cmp == 0:
                        eiej_cmp = compare_to_desc(int(ei), int(ej))

                    return eiej_cmp

                # Sort entities: first by score descending and, if tied, by degree ascending.
                selected_entities = sorted(list(corrupted_scores), key=cmp_to_key(compare_entities))

                def get_siminfo_to_print(selected_si):
                    if not isinstance(selected_si, dict):
                        return selected_si

                    if direction == 'head':
                        if selected_si['ri'] == r and selected_si['ri_type'] == 'domain' and selected_si['sim'] > 0:
                            return \
                                f'\t({selected_si['ri_type']}--{selected_si['rj_type']}) ' \
                                f'with predicate {main_loader.relation_to_id[selected_si['rj']]} ' \
                                f'and similarity {selected_si["sim"]}'

                        if selected_si['rj'] == r and selected_si['rj_type'] == 'domain' and selected_si['sim'] > 0:
                            return \
                                f'\t({selected_si['rj_type']}--{selected_si['ri_type']}) ' \
                                f'with predicate {main_loader.relation_to_id[selected_si['ri']]} ' \
                                f'and similarity {selected_si["sim"]}'
                    elif direction == 'tail':
                        if selected_si['ri'] == r and selected_si['ri_type'] == 'range' and selected_si['sim'] > 0:
                            return \
                                f'\t({selected_si['ri_type']}--{selected_si['rj_type']}) ' \
                                f'with predicate {main_loader.relation_to_id[selected_si['rj']]} ' \
                                f'and similarity {selected_si["sim"]}'
                        if selected_si['rj'] == r and selected_si['rj_type'] == 'range' and selected_si['sim'] > 0:
                            return \
                                f'\t({selected_si['rj_type']}--{selected_si['ri_type']}) ' \
                                f'with predicate {main_loader.relation_to_id[selected_si['ri']]} ' \
                                f'and similarity {selected_si["sim"]}'

                    return None

                # # Do not remove! This helps figure out relation compatibility.
                # print(f'Relation similarity for {main_loader.relation_to_id[r]} and direction: {direction}')
                # for sim_info in self.all_info_similarities:
                #     to_print = get_siminfo_to_print(sim_info)
                #     if to_print is not None:
                #         print(to_print)
                # # Done with relation compatibility.

                self.selected_entities_by_rd[r][direction] = {'entities': selected_entities}
                if corrupted_sim_info is not None:
                    self.selected_entities_by_rd[r][direction]['sim_info'] = corrupted_sim_info

                # Get the percentiles based on number of bins.
                step = 1/number_of_bins

                bins = {} # This is just for debugging!
                relevance_regular, relevance_reversed = \
                    np.zeros(len(selected_entities)), np.zeros(len(selected_entities))
                relevance, last_index = number_of_bins - 1, -1
                while relevance >= 0:
                    init, end = last_index + 1, \
                        min(int(last_index + 1 + step * len(selected_entities)) + 1, len(selected_entities))
                    bins[relevance] = (init, end)

                    relevance_regular[init:end] = relevance
                    relevance_reversed[init:end] = number_of_bins - relevance - 1
                    self.total_relevance_by_direction[direction][relevance] += (end - init)

                    last_index = end
                    relevance -= 1

                self.selected_entities_by_rd[r][direction]['regular'] = relevance_regular
                self.selected_entities_by_rd[r][direction]['reversed'] = relevance_reversed

                # # Do not remove! This helps find examples for predicate similarity.
                # current_k = 5
                # print(f'Top and bottom entities for {main_loader.relation_to_id[r]} and direction: {direction}')
                # for relevance in bins:
                #     (init, end) = bins[relevance]
                #
                #     print(f'\tRelevance {relevance}:')
                #
                #     last_siminfo, last_index = None, None
                #     siminfo_indexes = []
                #     for i in range(init, end):
                #         current_entity = selected_entities[i]
                #         current_siminfo = corrupted_sim_info[current_entity]
                #
                #         if last_siminfo is None:
                #             last_siminfo = current_siminfo
                #             last_index = i
                #
                #         if current_siminfo != last_siminfo:
                #             siminfo_indexes.append((last_siminfo, last_index, i))
                #             last_siminfo, last_index = current_siminfo, i
                #
                #     # Add last!
                #     siminfo_indexes.append((last_siminfo, last_index, i - 1))
                #
                #     for (siminfo, current_init, current_end) in siminfo_indexes:
                #         print(f'\tSimilarity {get_siminfo_to_print(siminfo)}:')
                #
                #         print(f'\t\tTop {current_k}:')
                #         for i in range(current_init, min(current_init + current_k, current_end)):
                #             if i >= len(selected_entities):
                #                 break
                #
                #             print(
                #                 f'\t\t\t{main_loader.entity_to_id[selected_entities[i]]}; '
                #                 f'predicate degree: {self.entity_degree_by_relation[r][selected_entities[i]]}; '
                #                 f'global degree: {self.entity_degree[selected_entities[i]]}; '
                #                 f'position: {i}'
                #             )
                #
                #         print(f'\t\tBottom {current_k}:')
                #         for i in range(max(current_init, current_end - current_k - 1), current_end):
                #             print(
                #                 f'\t\t\t{main_loader.entity_to_id[selected_entities[i]]}; '
                #                 f'predicate degree: {self.entity_degree_by_relation[r][selected_entities[i]]}; '
                #                 f'global degree: {self.entity_degree[selected_entities[i]]}; '
                #                 f'position: {i}'
                #             )
                # # Done finding examples for predicate similarity.

    def _aggregate_structures(self):
        """Precomputes the union of domains and relations from all provided loaders."""
        for loader in (self.main_loader, *self.secondary_loaders):
            for r in loader.head_dict:
                for t, heads in loader.head_dict[r].items():
                    self.head_dict[r][t].update(heads)

            for r in loader.tail_dict:
                for h, tails in loader.tail_dict[r].items():
                    self.tail_dict[r][h].update(tails)

            for r, dom_entities in loader.domain.items():
                self.domain[r].update(dom_entities)

            for r, range_entities in loader.range.items():
                self.range[r].update(range_entities)

        # self.dom_dom = self.main_loader.domDomCompatible
        # self.dom_ran = self.main_loader.domRanCompatible
        # self.ran_dom = self.main_loader.ranDomCompatible
        # self.ran_ran = self.main_loader.ranRanCompatible

    def _convert_to_numpy(self):
        """Converts sets in dictionaries to NumPy arrays for optimized operations."""
        for r in self.head_dict:
            for t in self.head_dict[r]:
                self.head_dict[r][t] = np.array(list(self.head_dict[r][t]))
        for r in self.tail_dict:
            for h in self.tail_dict[r]:
                self.tail_dict[r][h] = np.array(list(self.tail_dict[r][h]))

        # Convert these to numpy too to be used in corruption modes
        for r in self.domain:
            self.domain[r] = np.array(list(self.domain[r]))
        for r in self.range:
            self.range[r] = np.array(list(self.range[r]))

    def get_triples(self):
        """Returns all triples in the main data loader."""
        return self.main_loader.get_triples()

    def _build_compatible_relations(self):
        """
        Constructs a unified compatible_relations dictionary based on type of compatibility.

        :return: Dictionary {(r, elem_type): [(r', elem_type'), ...]}
        """
        compat = {}
        all_relations = set(self.dom_dom.keys()) | set(self.dom_ran.keys()) | set(self.ran_dom.keys()) | set(self.ran_ran.keys())

        for r in all_relations:
            domain_compat = []
            range_compat = []

            if r in self.dom_dom:
                domain_compat.extend([(r_prime, "domain") for r_prime in self.dom_dom[r]])
            if r in self.dom_ran:
                domain_compat.extend([(r_prime, "range") for r_prime in self.dom_ran[r]])
            if r in self.ran_dom:
                range_compat.extend([(r_prime, "domain") for r_prime in self.ran_dom[r]])
            if r in self.ran_ran:
                range_compat.extend([(r_prime, "range") for r_prime in self.ran_ran[r]])

            compat[(r, "domain")] = domain_compat
            compat[(r, "range")] = range_compat

        return compat

    def _get_elements(self, relation, elem_type):
        """
        Retrieves the set of elements (domain or range) for a given relation.

        :param relation: Relation ID or name
        :param elem_type: Either 'domain' or 'range'
        :return: Set of entities
        """
        if elem_type == "domain":
            return self.domain.get(relation, np.array([]))
        elif elem_type == "range":
            return self.range.get(relation, np.array([]))
        else:
            raise ValueError("elem_type must be 'domain' or 'range'")

    def _get_extended_elements(self, relation, elem_type):
        """
        Computes the union of compatible elements for one-hop extended corruption.

        :param relation: Relation ID or name
        :param elem_type: 'domain' or 'range'
        :return: Union of compatible elements (set)
        """
        extended = set()
        for (rel_prime, type_prime) in self.compatible_relations.get((relation, elem_type), []):
            elems = self._get_elements(rel_prime, type_prime)
            extended.update(elems)
        return np.array(list(extended))

    def get_corrupted(self, h, r, t, corruption_type='tail', corruption_mode='LCWA'):
        if corruption_mode == 'LCWA':
            main_set = self.entities
        else:
            if 'non' not in corruption_mode and 'sensical' in corruption_mode and corruption_type == 'tail':
                elem_type = 'range'
            elif 'non' not in corruption_mode and 'sensical' in corruption_mode and corruption_type == 'head':
                elem_type = 'domain'
            elif 'nonsensical' in corruption_mode and corruption_type == 'tail':
                elem_type = 'domain'
            elif 'nonsensical' in corruption_mode and corruption_type == 'head':
                elem_type = 'range'

            if 'one-hop' not in corruption_mode:
                main_set = self._get_elements(r, elem_type)
            else:
                main_set = self._get_extended_elements(r, elem_type)

        if corruption_type == 'tail':
            to_remove = self.tail_dict[r].get(h, np.empty(0))
        elif corruption_type == 'head':
            to_remove = self.head_dict[r].get(t, np.empty(0))

        return np.setdiff1d(main_set, to_remove, assume_unique=True)

