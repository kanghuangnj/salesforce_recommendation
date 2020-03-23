from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine

gmf_config = {'alias': 'gmf_factor32neg4-implict',
              'num_epoch': 200,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 39952,
              'num_items': 512,
              'latent_dim': 16,
            #   'acc_latent_dim': 879,
            #   'loc_latent_dim': 20,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
              'data_dir':'data/wework',
              'implicit': True,
              'negative_path': 'data/wework/loc_city_info.pkl'
            #   'pretrain_acc': 'data/wework/company.bin',
            #   'pretrain_loc': 'data/wework/location.bin',
              }

mlp_config = {'alias': 'mlp-implicit-negative1-sgd',
              'num_epoch': 50,
              'batch_size': 256,  # 1024,
              'optimizer': 'sgd',
              'adam_lr': 1e-3,
              'sgd_lr': .1,
              'sgd_momentum': 0.9,
              'num_users': 137601,  #39952,
              'num_items': 445, #83,    #512,
              'acc_latent_dim': 797, #741,
              'loc_latent_dim': 314,
              'num_negative': 1,
              'layers': [1111,256,64,32],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 1e-5,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain_mlp': 'checkpoints/{}'.format('mlp-explicit-negative1-sgd_Epoch4_HR0.5186_NDCG0.3254.model'),
              #'pretrain_mlp': 'checkpoints/{}'.format('mlp-explicit_Epoch22_HR0.5152_NDCG0.2955.model'),
              #'pretrain_mlp': 'checkpoints/{}'.format('mlp-implicit-negative1_Epoch4_HR0.5321_NDCG0.3179.model'),
              #'pretrain_mlp': 'checkpoints/{}'.format('mlp-explicit-negative_Epoch47_HR0.2572_NDCG0.1208.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
              'data_dir': 'data/wework',
              'train_filename': 'ratings_3rd.dat',
              'test_filename': 'ratings_test.dat',
              'implicit': True,
              'city_location_path': 'data/wework/loc_city_info.pkl',
              'covisit_location_path': 'data/wework/covisit.csv',
              'pretrain_acc': 'data/wework/company.npy',
              'pretrain_loc': 'data/wework/location.npy',}

neumf_config = {'alias': 'neumf_factor42neg4-implicit',
                'num_epoch': 100,
                'batch_size': 512,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 39952,
                'num_items': 512,
                'acc_latent_dim': 741,
                'loc_latent_dim': 20,
                'latent_dim_mf': 16,
                'latent_dim_mlp': 32,
                'num_negative': 4,
                'layers': [761,256,96,32,8],  # layers[0] is the concat of latent user vector & latent item vector
                'common_layers': [48, 8], 
                'l2_regularization': 1e-4,
                'use_cuda': False,
                'device_id': 0,
                'pretrain': False,
                'implicit': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor32neg4-implict_Epoch55_HR0.3451_NDCG0.1813.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor32neg4-explicit_Epoch74_HR0.4946_NDCG0.2642.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                'data_dir':'data/wework',
                'negative_path': 'data/wework/loc_city_info.pkl',
                'pretrain_acc': 'data/wework/company.npy',
                'pretrain_loc': 'data/wework/location.npy',
                }

MODEL_SPEC = {
  'gmf':{
    'config': gmf_config,
    'engine': GMFEngine,
  },
  'mlp':{
    'config': mlp_config,
    'engine': MLPEngine,
  },
  'neumf':{
    'config': neumf_config,
    'engine': NeuMFEngine
  }
}
