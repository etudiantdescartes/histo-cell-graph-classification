import pickle
import json
import os
from glob import glob
from tqdm import tqdm

def pickle_to_json(pickle_path, output_path, class_mapping):
    """
    Conversion of the output files from HoVer-UNet from pickle to json
    """
    with open(pickle_path, 'rb') as infile:
        dico = pickle.load(infile)
    
    #deleting useless information
    del dico['inst_map']
    del dico['inst_uid']
    del dico['inst_bbox']
    del dico['inst_type_prod']
    
    #from array to list before json creation
    dico['inst_type'] = dico['inst_type'].tolist()
    dico['inst_centroid'] = dico['inst_centroid'].tolist()
    dico['inst_contour'] = [contour.tolist() for contour in dico['inst_contour']]
    
    #renaming keys
    dico['type'] = dico.pop('inst_type')
    dico['centroid'] = dico.pop('inst_centroid')
    dico['contour'] = dico.pop('inst_contour')
    
    name = os.path.splitext(os.path.basename(pickle_path))[0] + '.json'
    
    #assigning graph class from the file name (0 or 1)
    for key in list(class_mapping.keys()):
        if key in name:
            dico['graph_class'] = class_mapping[key]
    
    output = os.path.join(output_path, name)
    with open(output, 'w') as json_file:
        json.dump(dico, json_file)
    


paths = glob('predictions/*.pickle')
output_dir = 'json'
class_mapping = {'_N_':0, '_PB_':0, '_UDH_':0, '_DCIS_':1, '_IC_':1,}
for path in tqdm(paths):
    pickle_to_json(path, output_dir, class_mapping)