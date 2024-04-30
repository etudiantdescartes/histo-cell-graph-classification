from torch_geometric.data import Data, Dataset
import torch
import os
import json
from scipy.spatial import Delaunay, distance_matrix
import cv2
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np



def graph_overlay(path, points, rng, graph_class):
    """
    Draw graphs over images for visualization, and saving them in 'overlay' folder
    """
    img_name = os.path.splitext(os.path.basename(path))[0] + '.jpg'
    img_path = f'C:/Users/scott/Desktop/cell_inference/overlay/{img_name}'
    img = cv2.imread(img_path)
    
    for edge in rng:
        pt1 = int(points[edge[0]][0]), int(points[edge[0]][1])
        pt2 = int(points[edge[1]][0]), int(points[edge[1]][1])
        cv2.line(img, pt1, pt2, (0,255,0), 2)
    
    output_name = f'C:/Users/scott/Desktop/cell_inference/graph_overlay/{img_name}'
    cv2.imwrite(output_name, img)


def relative_neighborhood_graph(points, dist_matrix):
    """
    Builds a RNG from Delaunay graph (for practical purposes)
    """
    triangulation = Delaunay(points)
    rng = []
    for simplex in triangulation.simplices:
        for i, j in [(0,1), (1,2), (0,2)]:
            pi, pj = simplex[i], simplex[j]
            edge_length = dist_matrix[pi, pj]
            if all(dist_matrix[pi][k] > edge_length or dist_matrix[pj][k] > edge_length for k in range(len(points)) if k != pi and k != pj):
                rng.append((pi, pj))
    return rng


def distance_threshold_graph(points, distance_threshold, dist_matrix):
    """
    Creates a graph by linking each node to every other node within a certain distance
    """
    edges = []
    for i in range(len(dist_matrix)):
        for j in range(i+1, len(dist_matrix)):
            if dist_matrix[i,j] < distance_threshold:
                edges.append((i,j))
    return edges
    

def graph_creation(path, nb_cell_type, draw_graph, distance_threshold, graph_type, include_edge_atributes, include_node_features):
    """
    Creates the graph (as a 'Data' object from PyG) with the nodes and edges embeddings
    """
    with open(path) as file:
        graph_info = json.load(file)
        
    points = graph_info['centroid']
    dist_matrix = distance_matrix(points, points)
    
    if graph_type == 'relative_neighborhood_graph':
        graph = relative_neighborhood_graph(points, dist_matrix)
    elif graph_type == 'distance_threshold_graph':
        graph = distance_threshold_graph(points, distance_threshold, dist_matrix)            
    
    graph_class = graph_info['graph_class']
    if draw_graph:
        graph_overlay(path, points, graph, graph_class)
    
    
    #Building a one-hot encoded vector for the cell type
    cell_type = graph_info['type']
    one_hot_cell_class = torch.zeros(len(cell_type), nb_cell_type, dtype=torch.float32)
    for i, t in enumerate(cell_type):
        one_hot_cell_class[i, t] = 1
        
    if include_node_features:
        #Concatenation with the rest of the features that are continuous values
        node_features = torch.cat((torch.tensor(graph_info['eccentricity']).unsqueeze(1),
                                   torch.tensor(graph_info['solidity']).unsqueeze(1),
                                   torch.tensor(graph_info['area']).unsqueeze(1),
                                   torch.tensor(graph_info['orientation']).unsqueeze(1),
                                   torch.tensor(graph_info['cell_perimeter']).unsqueeze(1),
                                   torch.tensor(graph_info['contrast']).unsqueeze(1),
                                   torch.tensor(graph_info['major_axis_length']).unsqueeze(1),
                                   torch.tensor(graph_info['minor_axis_length']).unsqueeze(1),
                                   one_hot_cell_class),
                                  dim=1)
                                   #torch.tensor(np.array(graph_info['centroid'])[:,0]).unsqueeze(1),
                                   #torch.tensor(np.array(graph_info['centroid'])[:,1]).unsqueeze(1),
    
    
        #Normalization of area, perimeter and axis lengths according to their distribution in the dataset
        for i in [2, 4, 6, 7]:
            col = node_features[:,i]
            min_value = col.min()
            max_value = col.max()
            normalized_column = (col - min_value) / (max_value - min_value)
            node_features[:, i] = normalized_column
        
    else:
        node_features = one_hot_cell_class    
    
    #Adding edges from the graph to the Data object (2 edges in both direction for unoriented graphs in PyG)
    edges = []
    edge_features = []
    edge_weights = []
    for edge in graph:
        edges.append([edge[0], edge[1]])
        edges.append([edge[1], edge[0]])
        dist = dist_matrix[edge[0], edge[1]] / distance_threshold
        edge_weights.append(dist)
        edge_weights.append(dist)
        
        if include_edge_atributes:
            #Normalizing distance in respect to the distance threshold for creating the graph (max distance)
            one_hot_edge_feature = [0 for i in range(nb_cell_type)]
            one_hot_edge_feature[cell_type[edge[0]]] = 1
            edge_features.append(one_hot_edge_feature)
            
            one_hot_edge_feature = [0 for i in range(nb_cell_type)]
            one_hot_edge_feature[cell_type[edge[1]]] = 1
            edge_features.append(one_hot_edge_feature)
            

        
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    edges = torch.tensor(edges)
    
    if include_edge_atributes:
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        graph_data = Data(x=node_features, edge_weight=edge_weights, edge_attr=edge_features, edge_index=edges.t().contiguous(), y=torch.tensor([graph_class]))
    else:
        graph_data = Data(x=node_features, edge_weight=edge_weights, edge_index=edges.t().contiguous(), y=torch.tensor([graph_class]))
    
    #Adding information later facilitate result visualization
    graph_data.metadata = {'coordinates': points, 'file_path': path}
    
    return graph_data
    
    
    
class GraphDataset(Dataset):
    """
    Simple Dataset class to format the graph list for the training
    """
    def __init__(self, graph_list, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]
    
    
def separate_classes(dataset):
    """
    Separate the graphs into one list for each class, and sorts them in decreasing order by their number of nodes
    """
    g0, g1 = [], []
    for graph in dataset:
        if graph.y == 0:
            g0.append(graph)
        else:
            g1.append(graph)

    g0 = sorted(g0, key=lambda graph: len(graph.x))
    g1 = sorted(g1, key=lambda graph: len(graph.x))

    return g0, g1


def train_val_test_split(graphs):
    """
    Splits the graphs into train, val and test sets while distributing evenly the classes and graph sizes
    """
    graph_sizes = [graph.num_nodes for graph in graphs]

    lower_threshold = np.percentile(graph_sizes, 33)
    upper_threshold = np.percentile(graph_sizes, 66)

    small_graphs = [graph for graph, size in zip(graphs, graph_sizes) if size < lower_threshold]
    medium_graphs = [graph for graph, size in zip(graphs, graph_sizes) if lower_threshold <= size < upper_threshold]
    big_graphs = [graph for graph, size in zip(graphs, graph_sizes) if size >= upper_threshold]

    #Split data list into train, val, test
    def split_graphs(graph_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        total_graphs = len(graph_list)
        indices = list(range(total_graphs))
        np.random.shuffle(indices)

        train_end = int(train_ratio * total_graphs)
        val_end = train_end + int(val_ratio * total_graphs)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_graphs = [graph_list[i] for i in train_indices]
        val_graphs = [graph_list[i] for i in val_indices]
        test_graphs = [graph_list[i] for i in test_indices]

        return train_graphs, val_graphs, test_graphs

    train_small, val_small, test_small = split_graphs(small_graphs)
    train_medium, val_medium, test_medium = split_graphs(medium_graphs)
    train_big, val_big, test_big = split_graphs(big_graphs)

    #Combine the sets
    train_set = train_small + train_medium + train_big
    val_set = val_small + val_medium + val_big
    test_set = test_small + test_medium + test_big

    return train_set, val_set, test_set


if __name__ == '__main__':
    
    paths = glob(r'json/*.json')
    nb_cell_type = 6
    draw_graph = False
    distance_threshold = 80
    include_edge_atributes = False
    include_node_features = False
    graph_type = 'distance_threshold_graph' #'relative_neighborhood_graph'
    cpu_cores = os.cpu_count()
    print(f'Maximum number of processes: {cpu_cores}')
    arg_list = [(path, nb_cell_type, draw_graph, distance_threshold, graph_type, include_edge_atributes, include_node_features) for path in paths]
    print('Creating graphs from json files...')
    
    #Multiprocessing to create multiple graphs at the same time (uses every CPU core by default, decrease cpu_cores if necessary)
    with Pool(cpu_cores) as p:
        graph_list = p.starmap(graph_creation, arg_list)
        
    print('Creating torch dataset...')
    g0, g1 = separate_classes(graph_list)
    train_set_g0, val_set_g0, test_set_g0 = train_val_test_split(g0)
    train_set_g1, val_set_g1, test_set_g1 = train_val_test_split(g1)
    train_dataset, val_dataset, test_dataset = GraphDataset(train_set_g0+train_set_g1), GraphDataset(val_set_g0+val_set_g1), GraphDataset(test_set_g0+test_set_g1)
    
    #Saving datasets in torch_dataset folder
    torch.save(train_dataset, 'torch_datasets/train_set.pt')
    torch.save(val_dataset, 'torch_datasets/val_set.pt')
    torch.save(test_dataset, 'torch_datasets/test_set.pt')
    print('Train, val, test datasets saved in torch_datasets folder')

    #Single process at a time (very slow)
    #for path in tqdm(paths):
    #    graph_creation(path, nb_cell_type, draw_graph, distance_threshold, graph_type, include_edge_atributes, include_node_features)

        
    
    