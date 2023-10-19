import numpy as np
from scipy.linalg import svd
from gensim.models import Word2Vec
from logging import getLogger
from corerec.utils.utils import set_color
from tqdm import tqdm

class SpecWalk(object):
    r"""
        global and local pattern extractor
    """
    def __init__(self, dataset, config):
        self.how = config.model.how
        self.n_users = dataset.user_num
        self.user_embeddings = None
        self.logger = getLogger()
        self._post_init(dataset, config)


    def _post_init(self, dataset, config):
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.spectra_dim = config.model.spectra_dim
        self.interaction_matrix = dataset.get_dense_interaction_mat()
        self.eigenvec_dict = {}
        self._svd()
        if self.how != 'svd':
            self.feature_dim = config.model.feature_dim
            self.graph = dataset.graph
            self.uid2node, self.node2uid = dataset.get_uid_node_maps()
            self.walks = self._generate_walks()

    def _svd(self):
        self.logger.info(set_color("Spectral Decomposition...", "blue"))
        U, S, Vt = svd(self.interaction_matrix, full_matrices=True)
        self.user_eigenvectors = U[:, :self.spectra_dim]
        self.item_eigenvectors = Vt.T[:, :self.spectra_dim]
        for u in range(self.n_users):
            self.eigenvec_dict[f"u_{u}"] = self.user_eigenvectors[u]
        for i in range(self.n_items):
            self.eigenvec_dict[f"i_{i}"] = self.item_eigenvectors[i]

    def _softmax(self, p):
        e_p = np.exp(p - np.max(p))
        return e_p / e_p.sum()

    #Spectral-guided random walk
    def _transition_probabilities(self, node, neighbors, eigenvectors):

        # probs = []
        # # node_id = self.node2uid[node]
        # for neighbor in neighbors:
        #     # neighbor_id = self.node2uid[neighbor]
        #     alignment = np.dot(eigenvectors[node], eigenvectors[neighbor])
        #     prob = alignment / np.sum([np.dot(eigenvectors[node], eigenvectors[n]) for n in neighbors])
        #     probs.append(prob)

        node_eigenv = eigenvectors[node]
        neighbor_eigenv = np.vstack(eigenvectors[neighbor] for neighbor in neighbors)
        alignments = np.dot(neighbor_eigenv, node_eigenv)

        normed_alignments = self._softmax(alignments)
        return normed_alignments

    def _biased_random_walk(self, graph, node, eigenvectors, walk_length=80):
        walk = [node]
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(walk[-1]))
            if len(neighbors) == 0:
                break
            probs = self._transition_probabilities(walk[-1], neighbors, eigenvectors)
            next_node = np.random.choice(neighbors, p=probs)
            walk.append(next_node)
        return walk

    def _generate_walks(self):
        walks = []
        for node in tqdm(self.graph.nodes(),
                         total=self.graph.number_of_nodes(),
                         desc=set_color("Generating walks", "pink")):
            # print(node)
            walks.append(self._biased_random_walk(self.graph, node, self.eigenvec_dict))
        return walks

    # def _guided_node2vec(self):
    #     walks = self._generate_walks()
    #     # Train node2vec model
    #     self.guided_node2vec = Word2Vec(walks, vector_size=self.feature_dim, window=5, min_count=0, sg=1, workers=4, epochs=1)

    def fit(self):
        if self.how == 'svd':
            self.user_embeddings = self.user_eigenvectors
        else:
            self.logger.info(set_color("Training node2vec...", "blue"))
            guided_node2vec = Word2Vec(self.walks, vector_size=self.feature_dim, window=5, min_count=0, sg=1, workers=4, epochs=1)

            self.logger.info(set_color("Building user embeddings...", "blue"))
            user_graph_embeds = []
            for uid in range(self.n_users):
                node = f"u_{int(uid)}"
                user_graph_embeds.append(guided_node2vec.wv[node])
            self.user_embeddings = np.vstack(user_graph_embeds)

            if self.how == 'both':
                user_spectral_embeds = self.user_eigenvectors
                self.user_embeddings = np.hstack((user_spectral_embeds, self.user_embeddings))
    def get_user_embeddings(self):
        return self.user_embeddings
