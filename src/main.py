import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from data import SampleGenerator
from config import MODEL_SPEC
from torch.optim.lr_scheduler import StepLR
from utils import plot_grad_flow

def flatten(df):
    negatives = []
    for index, row in df.iterrows():
      for negative in row['negatives']:
          negatives.append([row['account_id'], negative])
    return pd.DataFrame(negatives, columns=['account_id', 'negative_id'], dtype=np.int32)



def train(args):
    # Load Data
    spec = MODEL_SPEC[args.model]
    config = spec['config']
    config['pretrain'] = False
    wework_dir = config['data_dir']
    wework_rating = pd.read_csv(os.path.join(wework_dir,  config['train_filename']), sep=',', header=0, names=['','account_id', 'atlas_location_uuid', 'rating', 'timestamp', 'weight'],  engine='python')
    # # Reindex
    # account_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    # account_id['userId'] = np.arange(len(user_id))
    # ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    # item_id = ml1m_rating[['mid']].drop_duplicates()
    # item_id['itemId'] = np.arange(len(item_id))
    # ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    # ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of AccountId is [{}, {}]'.format(wework_rating.account_id.min(), wework_rating.account_id.max()))
    print('Range of LocationId is [{}, {}]'.format(wework_rating.atlas_location_uuid.min(), wework_rating.atlas_location_uuid.max()))
    
    Engine = spec['engine']
    # DataLoader for training
    sample_generator = SampleGenerator(wework_rating, config)
    sample_generator.test()
    test_data = sample_generator.evaluate_data
    sample_generator.val()
    val_data = sample_generator.evaluate_data

   
    # Specify the exact model
    engine = Engine(config)
    # gamma = decaying factor
    scheduler = StepLR(engine.opt, step_size=1, gamma=0.75)
    train_negatives = []
    best_epoch = 0
    best_metric = float('inf')
    HR_10, NDCG_10 = 0, 0
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)

        train_loader, train_negative = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        scheduler.step()
        plot_grad_flow(engine.model.named_parameters(), epoch)
        train_negative = flatten(train_negative)
        if len(train_negatives) != 0:
            train_negatives = pd.concat([train_negatives, train_negative], axis=0)
        else:
            train_negatives = train_negative
        metric, auc, HR5, HR10, NDCG5, NDCG10 = engine.evaluate(val_data, train_negatives, epoch_id=epoch)
        if metric < best_metric :
            best_epoch = epoch
            best_metric = metric
            HR_10, NDCG_10 = HR10, NDCG10
            engine.save(config['alias'], epoch, HR_10, NDCG_10)
            print ('Epoch {}: found best results on validation data: metric = {:.4f}, HR10 = {:.4f}, NDCG10 = {:.4f}'.format(epoch, best_metric, HR_10, NDCG_10))

    engine.load(config['alias'], best_epoch, HR_10, NDCG_10)
    metric, auc, HR5, HR10, NDCG5, NDCG10 = engine.evaluate(test_data, train_negatives, epoch_id=epoch)
    print('Best Epoch {}: metric = {:.4f}, auc = {:.4f}, HR@5 = {:.4f}, HR@10 = {:.4f},\
          NDCG@5 = {:.4f}, NDCG@10 = {:.4f}'.format(best_epoch, metric, auc, HR5, HR10, NDCG5, NDCG10))

def test(args):
    batch_size = 1024
    spec = MODEL_SPEC[args.model]
    config = spec['config']
    config['pretrain'] = True
    wework_dir = config['data_dir']
    test = pd.read_csv(os.path.join(wework_dir, config['test_filename']), sep=',', header=0, names=['', 'account_id', 'atlas_location_uuid'],  engine='python')
    spec = MODEL_SPEC[args.model]
    config = spec['config']
    Engine = spec['engine']
    # Specify the exact model
    engine = Engine(config)
    model = engine.model.eval()
    all_test_accs, all_test_locs, all_test_scores = [], [], []
    labels = []
    for row in test.itertuples():
        all_test_accs.append(int(row.account_id))
        all_test_locs.append(int(row.atlas_location_uuid))
        # labels.append(row.rating)
   
    for i in tqdm(range(0, len(all_test_accs) // batch_size+1)):
        test_accs = torch.LongTensor(all_test_accs[i*batch_size: min(len(all_test_accs), (i+1)*batch_size)])
        test_locs = torch.LongTensor(all_test_locs[i*batch_size: min(len(all_test_accs), (i+1)*batch_size)])
        if config['use_cuda'] is True:
            test_accs = test_accs.cuda()
            test_locs = test_locs.cuda()
        rating_logits = model(test_accs, test_locs)
        ratings = nn.Sigmoid(rating_logits )
        all_test_scores.extend(ratings.data.view(-1).tolist())
    
    test_results = {'account_id': all_test_accs, 
                    'atlas_location_uuid': all_test_locs,
                    'prob': all_test_scores,}
                    # 'label': labels}    
    pred_df = pd.DataFrame(test_results)
    pred_df.to_csv('pred_test_3rd_cls_negative1.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', choices=['gmf', 'mlp', 'neumf'])
    arg('--mode', choices=['train', 'test'])
    
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)