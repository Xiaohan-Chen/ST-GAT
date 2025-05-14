import os
import numpy as np
import torch
import wntr
import shutil
from torch_geometric.data import InMemoryDataset, Data

class WaterGraphDataset(InMemoryDataset):
    """
    The data and labels are stored in the form of numpy files.
    Attentions:
        1. The data files are named as 'data_0.csv', 'data_1.csv', ..., 'data_N.csv'.
        2. The labels are stored in the file named 'contamination_nodes.csv'.
        3. Simulation last for 8 days, and the data are sampled every 30 minutes.
        4. Contamination starts at the 5th day with random bias.
        5. Contamination lasts a maximum of 12 hours and a minimum of 1 hour.
        6. Contamination nodes are randomly selected.
        7. Data formate: [timestamp, sensor1, sensor2, ..., sensorN]
    """
    def __init__(self, args, normalize=True, transform=None, pre_transform=None, pre_filter=None, padding=True):
        self.normal_start = 3 * 24 * 2
        self.normal_end = 4 * 24 * 2
        self.contam_start = 5 * 24 * 2
        self.contam_end = 6 * 24 * 2
        self.normalize = normalize
        self.padding = padding
        self.network_path = os.path.join('./Networks', args.network + '.inp')

        # Load water network FIRST
        try:
            self.wn = wntr.network.WaterNetworkModel(self.network_path)
            print('Successfully loaded the .inp file')
            self.node_name_list = self.wn.node_name_list
            self.edge_name_list = self.wn.link_name_list
        except FileNotFoundError:
            raise FileNotFoundError(f'Error: the file {self.network_path} was not found')
        except Exception as e:
            raise RuntimeError(f'Error loading the network: {e}')
        self.root = os.path.join(args.datadir, args.network)
        if os.path.isdir(self.root+'/processed'):
            shutil.rmtree(self.root+'/processed')
        self.datalist = self._load_file(self.root)
        self.contam_nodes = np.loadtxt(os.path.join(self.root, 'contamination_nodes.csv'), 
                                      dtype=str, delimiter=',')
        self.sensor_positions = np.loadtxt(os.path.join(self.root, 'sensor_positions.csv'), 
                                         dtype=str, delimiter=',')
        self.selected_sensors = args.sensor_id
        self.num_normal = int(np.mean(np.unique(self.contam_nodes, return_counts=True)[1]))

        # Get graph properties
        self.G = self.wn.to_graph()
        self.is_directed = self.G.is_directed()
        self.number_of_nodes = self.G.number_of_nodes()
        self.number_of_edges = self.G.number_of_edges()
        self.edge = [[start, end] for start, end, _ in self.G.edges]

        # get edge lengths and diameters
        self.edge_lengths = torch.tensor(self.wn.query_link_attribute('length').values,
                                         dtype=torch.float32)
        self.edge_dias = torch.tensor(self.wn.query_link_attribute('diameter').values, 
                                      dtype=torch.float32)

        self.data, self.labels, self.node_to_idx, self.idx_to_node = self._load_data()
        # Initialize parent class and load data
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _load_file(self, path):
        datalist = [f for f in os.listdir(path) if f.startswith('data')]
        return datalist

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _label_mapping(self, contam_nodes):
        """
        Map contamination nodes to indices
        Returns:
            idx_nodes: Indices of contamination nodes
            node_to_idx: {node_name: index}
            idx_to_node: {index: node_name}
        """
        node_to_idx = {node: idx for idx, node in enumerate(self.node_name_list)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        idx_nodes = [node_to_idx[node] for node in contam_nodes]
        return np.array(idx_nodes, dtype=np.int64).reshape(-1,1), node_to_idx, idx_to_node

    def _fetch_selecged_sensor_indices(self):
        return [np.where(self.sensor_positions == sensor)[0][0] for sensor in self.selected_sensors]

    def _load_data(self):
        """Load and process all data files"""
        normal_data, contam_data = [], []
        
        for i, filename in enumerate(self.datalist):
            filepath = os.path.join(self.root, filename)
            reading = np.loadtxt(filepath, delimiter=',')
            fetched_reading_idx = self._fetch_selecged_sensor_indices()
            reidx_fetched_reading = [idx + 1 for idx in fetched_reading_idx]
            
            if i < self.num_normal:
                normal_data.append(reading[self.normal_start:self.normal_end, reidx_fetched_reading].T) # [num_sensors, num_timesteps]
            contam_data.append(reading[self.contam_start:self.contam_end, reidx_fetched_reading].T) # [num_sensors, num_timesteps]
        
        # Stack and convert data
        normal_data = np.array(normal_data, dtype=np.float32)
        contam_data = np.array(contam_data, dtype=np.float32)
        data = np.vstack([contam_data, normal_data])
        
        # Create labels
        normal_labels = np.ones((self.num_normal, 1), dtype=np.int64) * self.number_of_nodes
        contam_labels, node_to_idx, idx_to_node = self._label_mapping(self.contam_nodes)
        labels = np.vstack([contam_labels, normal_labels])

        data_tensor = torch.tensor(data)
        labels_tensor = torch.tensor(labels)

        # Shuffle data and labels
        idx = torch.randperm(len(data_tensor))
        data_tensor = data_tensor[idx]
        labels_tensor = labels_tensor[idx]
        return data_tensor, labels_tensor, node_to_idx, idx_to_node

    def _get_sensor_idx(self):
        """Get indices of sensor nodes"""
        return [self.node_to_idx[sensor] for sensor in self.selected_sensors]

    def _get_edge_indices(self):
        """Convert edge list to PyG COO format"""
        node_index_dict = self.node_to_idx
        edge_indices = [[node_index_dict[start], node_index_dict[end]] 
                       for start, end in self.edge]
        return torch.tensor(edge_indices, dtype=torch.long).t()

    def process(self):
        """Convert raw data to PyG Data objects"""
        sensor_idx = self._get_sensor_idx()
        edge_indices = self._get_edge_indices()
        data_list = []
        for i in range(self.data.shape[0]):
            x = self.data[i]  # [num_sensors, num_features]
            y = self.labels[i]
            if self.normalize:
                x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
            # Create full feature matrix
            if self.padding:
                full_x = torch.zeros(self.number_of_nodes, x.shape[1])
                full_x[sensor_idx] = x
            else:
                full_x = x
            data_list.append(Data(x=full_x, edge_index=edge_indices, y=y))
        torch.save(self.collate(data_list), self.processed_paths[0])