import numpy as np
import json


class CompatibleRelationsGenerator:
    def __init__(self, headDict, tailDict, domain, range, threshold=0.75):
        """
        Initialize with precomputed domain, range, headDict, and tailDict.

        Args:
            headDict (dict): {relation: {tail: set(heads)}}
            tailDict (dict): {relation: {head: set(tails)}}
            domain (dict): {relation: set of head entities}
            range (dict): {relation: set of tail entities}
        """
        self.headDict = headDict
        self.tailDict = tailDict
        self.domain = domain
        self.range = range
        self.threshold = threshold

        self.domDomCompatible = {}
        self.domRanCompatible = {}
        self.ranDomCompatible = {}
        self.ranRanCompatible = {}
        self.all_info_similarities = None

    def compute_compatible_relations(self, threshold=75, method="overlap", alpha=0.5, beta=0.5):
        """
        Compute compatible relations based on entity overlaps.

        Args:
            threshold (float): Overlap coefficient threshold for compatibility.
            method (str): Similarity measure to use (options: "overlap", "jaccard",
                          "dice", "cosine", "tversky").
            alpha (float): Tversky parameter for weighting A-B.
            beta (float): Tversky parameter for weighting B-A.
        """
        relation_list = list(self.domain.keys())

        for r in relation_list:
            self.domDomCompatible[r] = []
            self.domRanCompatible[r] = []
            self.ranDomCompatible[r] = []
            self.ranRanCompatible[r] = []

        # Convert dictionary sets to NumPy arrays for faster computations
        domain_arrays = {r: np.array(list(self.domain[r])) for r in self.domain}
        range_arrays = {r: np.array(list(self.range[r])) for r in self.range}

        # Let's compute and store all the similarities.
        self.all_info_similarities = []
        all_similarities = []
        for i in range(0, len(relation_list)):
            ri = relation_list[i]

            for j in range(i+1, len(relation_list)):
                rj = relation_list[j]

                # Compute and store similarities.
                sim = self._compute_similarity(domain_arrays[ri], domain_arrays[rj], method, alpha, beta)
                self.all_info_similarities.append(
                    {'ri': ri, 'ri_type': 'domain', 'rj': rj, 'rj_type': 'domain', 'sim': sim})
                all_similarities.append(sim)

                sim = self._compute_similarity(range_arrays[ri], range_arrays[rj], method, alpha, beta)
                self.all_info_similarities.append(
                    {'ri': ri, 'ri_type': 'range', 'rj': rj, 'rj_type': 'range', 'sim': sim})
                all_similarities.append(sim)

                sim = self._compute_similarity(domain_arrays[ri], range_arrays[rj], method, alpha, beta)
                self.all_info_similarities.append(
                    {'ri': ri, 'ri_type': 'domain', 'rj': rj, 'rj_type': 'range', 'sim': sim})
                all_similarities.append(sim)

                sim = self._compute_similarity(range_arrays[ri], domain_arrays[rj], method, alpha, beta)
                self.all_info_similarities.append(
                    {'ri': ri, 'ri_type': 'range', 'rj': rj, 'rj_type': 'domain', 'sim': sim})
                all_similarities.append(sim)

        # Let's choose the similarity threshold based on percentiles.
        similarity_threshold = np.percentile(all_similarities, threshold)

        for sim_info in self.all_info_similarities:
            if sim_info['sim'] >= similarity_threshold:
                if sim_info['ri_type'] == 'domain' and sim_info['rj_type'] == 'domain':
                    self.domDomCompatible[sim_info['ri']].append(sim_info['rj'])
                    self.domDomCompatible[sim_info['rj']].append(sim_info['ri'])

                if sim_info['ri_type'] == 'range' and sim_info['rj_type'] == 'range':
                    self.ranRanCompatible[sim_info['ri']].append(sim_info['rj'])
                    self.ranRanCompatible[sim_info['rj']].append(sim_info['ri'])

                if sim_info['ri_type'] == 'domain' and sim_info['rj_type'] == 'range':
                    self.domRanCompatible[sim_info['ri']].append(sim_info['rj'])
                    self.ranDomCompatible[sim_info['rj']].append(sim_info['ri'])

                if sim_info['ri_type'] == 'range' and sim_info['rj_type'] == 'domain':
                    self.ranDomCompatible[sim_info['ri']].append(sim_info['rj'])
                    self.domRanCompatible[sim_info['rj']].append(sim_info['ri'])

    # ---------------------------------------------------------------
    # Overlap Coefficient:
    # The overlap coefficient measures the similarity between two sets
    # based on their intersection size relative to the smaller set.
    #
    # Formula:
    #     Overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    #
    # Where:
    #     - A and B are two sets.
    #     - |A ∩ B| is the number of common elements.
    #     - min(|A|, |B|) ensures normalization based on the smaller set.
    #
    # This metric ranges from 0 (no overlap) to 1 (complete subset match).
    #
    # References:
    # - Wikipedia: https://en.wikipedia.org/wiki/Overlap_coefficient
    #
    # ---------------------------------------------------------------
    def _compute_overlap(self, array1, array2):
        """ Compute overlap coefficient between two NumPy arrays. """
        if array1.size == 0 or array2.size == 0:
            return 0.0
        intersection = np.intersect1d(array1, array2).size
        return intersection / min(array1.size, array2.size)

    def _compute_similarity(self, array1, array2, method="overlap", alpha=0.5, beta=0.5):
        """
        Compute similarity between two NumPy arrays using different methods.

        Args:
            array1 (np.array): First entity set.
            array2 (np.array): Second entity set.
            method (str): Similarity measure ("overlap", "jaccard", "dice", "cosine", "tversky").
            alpha (float): Tversky parameter for weighting A-B.
            beta (float): Tversky parameter for weighting B-A.

        Returns:
            float: Similarity score (0 to 1).
        """
        if array1.size == 0 or array2.size == 0:
            return 0.0

        intersection_size = np.intersect1d(array1, array2).size
        len_a = array1.size
        len_b = array2.size
        union_size = len_a + len_b - intersection_size

        # ---------------------------------------------------------------
        # 1. Overlap Coefficient:
        # Measures how much one set is a subset of another.
        # Formula: Overlap(A, B) = |A ∩ B| / min(|A|, |B|)
        # Reference: https://en.wikipedia.org/wiki/Overlap_coefficient
        # ---------------------------------------------------------------
        if method == "overlap":
            return intersection_size / min(len_a, len_b)

        # ---------------------------------------------------------------
        # 2. Jaccard Similarity:
        # Measures similarity considering both intersection and union.
        # Formula: Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        # Reference: https://en.wikipedia.org/wiki/Jaccard_index
        # ---------------------------------------------------------------
        elif method == "jaccard":
            return intersection_size / union_size

        # ---------------------------------------------------------------
        # 3. Sørensen-Dice Coefficient (Dice Similarity):
        # Weighs intersection more heavily.
        # Formula: Dice(A, B) = (2 * |A ∩ B|) / (|A| + |B|)
        # Reference: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # ---------------------------------------------------------------
        elif method == "dice":
            return (2 * intersection_size) / (len_a + len_b)

        # ---------------------------------------------------------------
        # 4. Cosine Similarity:
        # Treats sets as vectors in a multi-dimensional space.
        # Formula: Cosine(A, B) = |A ∩ B| / (sqrt(|A|) * sqrt(|B|))
        # Reference: https://en.wikipedia.org/wiki/Cosine_similarity
        # ---------------------------------------------------------------
        elif method == "cosine":
            return intersection_size / (np.sqrt(len_a) * np.sqrt(len_b))

        # ---------------------------------------------------------------
        # 5. Tversky Index:
        # A generalized form allowing asymmetric weighting.
        # Formula: Tversky(A, B) = |A ∩ B| / (|A ∩ B| + α |A - B| + β |B - A|)
        # Reference: https://en.wikipedia.org/wiki/Tversky_index
        # ---------------------------------------------------------------
        elif method == "tversky":
            a_minus_b = len_a - intersection_size
            b_minus_a = len_b - intersection_size
            return intersection_size / (intersection_size + alpha * a_minus_b + beta * b_minus_a)

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def save_to_file(self, file_path="compatible_relations.txt"):
        """ Save the computed compatible relations to a file. """
        with open(file_path, "w") as file:
            # json.dump(self.domDomCompatible, file)
            file.write(str(dict(sorted(self.domDomCompatible.items()))) + "\n")
            # json.dump(self.domRanCompatible, file)
            file.write(str(self.domRanCompatible) + "\n")
            # json.dump(self.ranDomCompatible, file)
            file.write(str(self.ranDomCompatible) + "\n")
            # json.dump(self.ranRanCompatible, file)
            file.write(str(self.ranRanCompatible) + "\n")

    def load_from_file(self, file_path="compatible_relations.txt"):
        """ Load the compatible relations from a file. """
        with open(file_path, "r") as file:
            self.domDomCompatible = json.loads(file.readline())
            self.domRanCompatible = json.loads(file.readline())
            self.ranDomCompatible = json.loads(file.readline())
            self.ranRanCompatible = json.loads(file.readline())

    def generate(self):
        """ Main function to compute and save compatible relations. """
        self.compute_compatible_relations(self.threshold)
        self.save_to_file()


class Triple():
    """
        Triple is the class that contains functionality for triples
    """
    __slots__ = "h", "r", "t"

    def __init__(self, h, r, t):
        """
            Initialise a triples
        Args:
            h (int): Head entity
            r (int): Relation entity
            t (int): Tail entity
        """
        self.h = h
        self.r = r
        self.t = t

    def __eq__(self, other):
        """
            Checks for equality between two triples
        Args:
            other (Triple): Triple object to be compared to

        Returns:
            bool: Returns true if triples are equal, false otherwise
        """
        return (self.h, self.r, self.t) == (other.h, other.r, other.t)

    def __str__(self):
        """
        Converts triple into a string

        Returns:
            str: [h,r,t]
        """
        return "[" + str(self.h) + "," + str(self.r) + "," + str(self.t) + "]"

    def __hash__(self):
        return int(self.pi(self.pi(self.h, self.r), self.t))

    def __str__(self):
        return str(self.h) + " " + str(self.r) + " " + str(self.t)

    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    def pi(self, k1, k2):
        return .5 * (k1 + k2) * (k1 + k2 + 1) + k2
