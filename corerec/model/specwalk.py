import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd
from gensim.models import Word2Vec
from logging import getLogger
from corerec.utils.utils import set_color
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from threading import Thread
from time import sleep

class SpecWalk(nn.Module):
    r"""
        global and local pattern extractor
    """
    def __init__(self, dataset, config):
        super(SpecWalk, self).__init__()
        self.how = config.model.how
        self.random_seed = config.dss_args.seed
        self.n_users = dataset.user_num
        self.user_embeddings = None
        self.logger = getLogger()
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.spectra_dim = config.model.spectra_dim
        self.interaction_matrix = dataset.get_dense_interaction_mat()
        self.eigenvec_dict = {}
        self.user_spec_embedding = nn.Embedding(self.n_users, embedding_dim=self.spectra_dim)
        self._svd()
        if self.how != 'svd':
            self.feature_dim = config.model.feature_dim
            self.user_walk_embedding = nn.Embedding(self.n_users, embedding_dim=self.feature_dim)
            self.graph = dataset.graph
            self.uid2node, self.node2uid = dataset.get_uid_node_maps()
            self.walks = self._generate_walks()
        # self._post_init(dataset, config)

    def _svd(self):
        self.logger.info(set_color("Spectral Decomposition...", "blue"))
        U, S, Vt = svd(self.interaction_matrix, full_matrices=True)
        self.user_eigenvectors = U[:, :self.spectra_dim]
        self.item_eigenvectors = Vt.T[:, :self.spectra_dim]
        for u in range(self.n_users):
            self.eigenvec_dict[f"u_{u}"] = self.user_eigenvectors[u]
        for i in range(self.n_items):
            self.eigenvec_dict[f"i_{i}"] = self.item_eigenvectors[i]
        self.user_spec_embedding.weight.data = torch.FloatTensor(self.user_eigenvectors)

    def _softmax(self, p):
        e_p = np.exp(p - np.max(p))
        return e_p / e_p.sum()

    #Spectral-guided random walk
    def _transition_probabilities(self, node, neighbors, eigenvectors):

        node_eigenv = eigenvectors[node]
        neighbor_eigenv = np.vstack([eigenvectors[neighbor] for neighbor in neighbors])
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

    # def _worker(self, nodes, progress_list):
    #     walks = []
    #     for node in nodes:
    #         walks.append(self._biased_random_walk(self.graph, node, self.eigenvec_dict))
    #         progress_list.append(1)  # Update the shared list to indicate progress
    #     return walks
    #
    # def _generate_walks(self):
    #     # Determine the number of available CPU cores and create a process pool
    #     num_processes = cpu_count()
    #     pool = Pool(processes=num_processes)
    #
    #     # Distribute the nodes across the processes
    #     total_nodes = list(self.graph.nodes())
    #     chunk_size = len(total_nodes) // num_processes
    #     chunks = [total_nodes[i:i + chunk_size] for i in range(0, len(total_nodes), chunk_size)]
    #
    #     # Create a shared list to track progress across processes
    #     manager = Manager()
    #     progress_list = manager.list()
    #
    #     # Function to update tqdm progress bar
    #     def update_progress(total):
    #         pbar = tqdm(total=total, desc="Generating walks")
    #         while len(progress_list) < total:
    #             pbar.update(len(progress_list) - pbar.n)  # Update the progress bar based on completed tasks
    #             sleep(0.1)  # Sleep for a short duration before checking again
    #         pbar.close()
    #
    #     # Start the tqdm updater in a separate thread
    #     thread = Thread(target=update_progress, args=(len(total_nodes),))
    #     thread.start()
    #
    #     # Use map to generate walks in parallel
    #     results = pool.starmap(self._worker, [(chunk, progress_list) for chunk in chunks])
    #
    #     # Close the pool and wait for the tasks to complete
    #     pool.close()
    #     pool.join()
    #
    #     # Wait for the tqdm thread to finish
    #     thread.join()
    #
    #     # Flatten the list of walks
    #     walks = [walk for sublist in results for walk in sublist]
    #
    #     return walks

    def fit(self):
        if self.how == 'svd':
            self.user_embeddings = self.user_spec_embedding
        else:
            self.logger.info(set_color("Training node2vec...", "blue"))
            guided_node2vec = Word2Vec(self.walks, vector_size=self.feature_dim,
                                       window=5, min_count=0,
                                       sg=1, workers=4, seed=self.random_seed)

            self.logger.info(set_color("Building user embeddings...", "blue"))
            user_graph_embeds = []
            for uid in range(self.n_users):
                node = f"u_{int(uid)}"
                user_graph_embeds.append(guided_node2vec.wv[node])
            self.user_walk_embedding.weight.data = torch.FloatTensor(np.vstack(user_graph_embeds))
            if self.how == 'both':
                self.user_embeddings = torch.cat((self.user_spec_embedding.weight, self.user_walk_embedding.weight), dim=1)
            else:
                self.user_embeddings = self.user_walk_embedding
    def get_all_user_embeddings(self):
        return self.user_embeddings
