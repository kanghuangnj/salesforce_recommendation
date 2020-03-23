import torch
from torch import nn
import numpy as np
from engine import Engine
from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        # self.latent_dim = config['latent_dim']
        acc_latent_dim = config['acc_latent_dim']
        loc_latent_dim = config['loc_latent_dim']
        self.embedding_account = nn.Embedding(num_embeddings=self.num_users, embedding_dim=acc_latent_dim)
        self.embedding_location = nn.Embedding(num_embeddings=self.num_items, embedding_dim=loc_latent_dim)
        self.embedding_id = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.num_items)

        if not self.config['pretrain']:
            self.init_weight()

        1111,256,64,32
        self.net_interaction = nn.Sequential(
            nn.Linear(acc_latent_dim+loc_latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.Linear(64, 16),
            # nn.BatchNorm1d(16),
            # nn.Dropout(p=0.2)
            # nn.LeakyReLU(),
        )
        
        self.net_id = nn.Sequential(
            nn.Linear(self.num_items, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            # nn.Linear(32, 8),
        )
        
        self.net_shared = nn.Sequential(
            nn.Linear(64+32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, out_features=1)
        )
        

        

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_account(user_indices)
        item_embedding = self.embedding_location(item_indices)
        id_embedding = self.embedding_id(item_indices) 
        vector_account_location = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        
        vector_id = self.net_id(id_embedding)
        vector_account_location = self.net_interaction(vector_account_location)
        vector = torch.cat([vector_account_location, vector_id], dim=-1)
        logits = self.net_shared(vector)
        #rating = self.logistic(logits)
        return logits

    def init_weight(self):
        acc_embedding = np.load(self.config['pretrain_acc'])
        loc_embedding = np.load(self.config['pretrain_loc'])
        print (acc_embedding.shape, loc_embedding.shape)
        self.embedding_account.weight = nn.Parameter(torch.FloatTensor(acc_embedding))
        self.embedding_location.weight = nn.Parameter(torch.FloatTensor(loc_embedding))
        self.embedding_account.require = False
        self.embedding_location.require = False
        self.embedding_id.require = True


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        if config['pretrain']:
            #self.model.load_pretrain_weights()
            resume_checkpoint(self.model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])
        super(MLPEngine, self).__init__(config)
        print(self.model)

       