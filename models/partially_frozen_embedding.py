import torch
import torch.nn as nn  

# a partially trainable, partially froezen, embedding
# pretained are the pretained weights, and trainable are  the number of embeddings to add to that that are trainable
# trainable indices follow the pretained ones. So if we pass in 10 pretrained embeddings, and ask for 10 trainable ones
# indices 0-9 are pretrained and frozen, 10-19 new and trainable.


class PartiallyFrozenEmbedding:
    def __init__(self, pretained,trainable):
        self.pretrained_embedding = nn.Embedding.from_pretrained(pretained,freeze=True)
        self.trainable_embedding = nn.Embedding(trainable, pretained.shape[1])
        
    def embed(self, x):
        mask = x >= self.pretrained_embedding.weight.shape[0]
        pretrained_x = x.clone()
        pretrained_x[mask] = 0

        embedded_x = self.pretrained_embedding(pretrained_x)

        # Every token without representation has to be brought into appropriate range
        x = x - self.pretrained_embedding.weight.shape[0]
        # Zero out the ones which already have pretrained embedding
        x[~mask] = 0

        non_pretrained_embedded_x = self.trainable_embedding(x)

        # And finally change appropriate tokens from placeholder embedding created by
        # pretrained into trainable embeddings.
        embedded_x[mask] = non_pretrained_embedded_x[mask]
        
        return embedded_x
        
    def to(self, device):
        self.pretrained_embedding.to(device)
        self.trainable_embedding.to(device)        