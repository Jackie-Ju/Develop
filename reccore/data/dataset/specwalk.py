import numpy as np
import networkx as nx
from tqdm import tqdm
from reccore.data.dataset.dataset import BasicDataset
from reccore.utils.utils import set_color
tqdm.pandas()

class DataSpecWalk(BasicDataset):
    def __init__(self, config=None):
        super(DataSpecWalk, self).__init__(config)
        self._load_data()
        self._build_graph()
        self.interaction_mat = self._build_sparse_matrix(dataset=self.dataset,
                                                         form='coo',
                                                         explicit=True,
                                                         shape=(self.user_num, self.item_num),
                                                         core=False)
        self._uid_node_mapping()

    def _build_graph(self):
        self.logger.info(set_color("Building user item bipartite graph...", "blue"))
        G = nx.Graph()
        # Add nodes
        G.add_nodes_from([f"u_{int(userId)}" for userId in self.dataset.uid.values], bipartite=0)
        G.add_nodes_from([f"i_{int(itemId)}" for itemId in self.dataset.iid.values], bipartite=1)

        # Add weights for edges
        G.add_weighted_edges_from([(f"u_{int(userId)}", f"i_{int(itemId)}", rating) for (userId, itemId, rating)
                                   in self.dataset[['uid', 'iid', 'rating']].values])
        self.G = G

    def _uid_node_mapping(self):
        self.node2uid = {f"u_{uid}":uid for uid in range(self.user_num)}
        self.uid2node = {uid:f"u_{uid}" for uid in range(self.user_num)}

    def __getitem__(self, index):
        if self.flag == 'train':
            return {"userId": 0}
        elif self.flag == 'valid':
            return {"userId": 0,
                    "indices": ([0]*1, [0]*1)}
        else:
            return {"userId": 0,
                    "indices": ([0]*1, [0]*1)}

    def __len__(self):
        if self.flag == 'train':
            return 1
        elif self.flag == 'valid':
            return 1
        else:
            return 1

    def get_uid_node_maps(self):
        return self.uid2node, self.node2uid

    @property
    def graph(self):
        return self.G

    def get_dense_interaction_mat(self):
        return self.interaction_mat.toarray()
