import torch
import torch.nn
from torch import nn
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        acc_latent_dim_mlp = config['acc_latent_dim']
        loc_latent_dim_mlp = config['loc_latent_dim']
        self.embedding_account_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=acc_latent_dim_mlp)
        self.embedding_location_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=loc_latent_dim_mlp)
        self.embedding_account_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_location_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        self.common_layers = torch.nn.Sequential(
            torch.nn.Linear(config['common_layers'][0], config['common_layers'][1]),
            nn.LeakyReLU()) 

        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-2], config['layers'][1:-1])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['common_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_account_mlp(user_indices)
        item_embedding_mlp = self.embedding_location_mlp(item_indices)
        user_embedding_mf = self.embedding_account_mf(user_indices)
        item_embedding_mf = self.embedding_location_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.LeakyReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        vector = self.common_layers(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""
        config = self.config
        mlp_model = MLP(config)
        device_id = -1
        if config['use_cuda'] is True:
            mlp_model.cuda()
            device_id = config['device_id']
        resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=device_id)

        self.embedding_account_mlp.weight.data = mlp_model.embedding_account.weight.data
        self.embedding_location_mlp.weight.data = mlp_model.embedding_location.weight.data

        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=device_id)
        self.embedding_account_mf.weight.data = gmf_model.embedding_account.weight.data
        self.embedding_location_mf.weight.data = gmf_model.embedding_location.weight.data

        self.embedding_account_mlp.require = False
        self.embedding_location_mlp.require = False
        self.embedding_account_mf.require = False
        self.embedding_location_mf.require = False
        # self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        # self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)


class NeuMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = NeuMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()
