This project aims at trying to classify histological images in one of two classes: begnin and malignant tumors, by representing them as graphs with cells as nodes, and classifying them with a graph neural network.
The information encoded into each of these node can be color or morphological-based characteristics.
Once the nodes are created, several graph construction methods can be used, the most common ones are the the relative neighborhood graph (a subgraph of Delau-
nay triangulation) and a simple distance threshold, or radius graph, consisting of linking each node to every other node within a certain distance.

# Installation

## Prerequisites
First install the required libraries:
- ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
- ```pip install torch_geometric```
- ```python pip install scikit-image```
- ```python pip install scipy```

## Running the code
The paths and parameters are hardcoded, change them accordingly.

- ```conversion.py``` converts the output files from hover-unet to json and easily readable files
- ```feature_extraction.py``` extracts color and morphological-based cell characteristics and adds them to the json files
- ```preprocessing.py``` converts the json files into graph datasets (train, test, val) and saves them
- ```model.py``` contains the GNN classification model
- ```train.py``` trains the model
- ```gnn_explainer.py``` unfinished
