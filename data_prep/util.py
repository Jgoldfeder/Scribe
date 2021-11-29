import pickle,torch,os
     
path_to_files = "resources" + os.sep
     
def save_tensor(tensor, name):
    torch.save(tensor,path_to_files + name)
    
def load_tensor(name):
    return torch.load(path_to_files + name)
    
    
def save_obj(obj, name ):
    with open(path_to_files + name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(path_to_files + name, 'rb') as f:
        return pickle.load(f)
