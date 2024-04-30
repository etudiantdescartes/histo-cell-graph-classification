from torch_geometric.explain import Explainer, GNNExplainer
import torch
import os
from preprocessing import GraphDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.load('torch_datasets/test_set.pt')
graph = dataset[0]
graph = graph.to(device)

model = torch.load('model/gnn_best_model.pt')
model = model.to(device)
model.eval()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=2000),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='probs',
    ),
)

explanation = explainer(graph.x, graph.edge_index, target=graph.y)
print(os.path.basename(graph.metadata['file_path']).split('.')[0])
print(explanation.node_mask, explanation.edge_mask)

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f'Explanation saved as {path}')