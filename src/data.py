import torch
import random
import pandas as pd
import pickle
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
random.seed(0)
class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, config):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'account_id' in ratings.columns
        assert 'atlas_location_uuid' in ratings.columns
        assert 'rating' in ratings.columns
        self.mode = None
        self.ratings = ratings
        if not config['implicit']:
        # explicit feedback using _normalize and implicit using _binarize
            self.preprocess_ratings = self._normalize(ratings)
        else:
            self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['account_id'].unique())

        # self.mode = 'mlp'
        # if negative path exist, it will select negative location from individual city
        # negative_path = None
        # if 'negative_path' in config:
        #     negative_path = config['negative_path']
        # if negative_path: 
        with open(config['city_location_path'], 'rb') as f:
            self.item_pool = pickle.load(f)
        self.covisit_df = pd.read_csv(config['covisit_location_path'])
        # if not it will select negative locations/accounts from all candidates
        # else:
        #     self.mode = 'mf'
        #     self.item_pool = set(self.ratings['location_id'].unique())
        # create negative item samples for NCF learning
       
        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.testval_ratings = self._split_loo(self.preprocess_ratings)
        self.test_ratings = self.testval_ratings.sample(len(self.testval_ratings) // 2).reset_index(drop=True)
        self.val_ratings = self.testval_ratings[~self.testval_ratings.index.isin(self.test_ratings.index)]

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max() + 5
        ratings['rating'] = (5+ratings.rating) * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, implicit feedback"""
        ratings = deepcopy(ratings)
        ratings.loc[ratings['rating'] > 0,'rating'] = 1.0
        #print (ratings['rating'].unique())
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['account_id'])['timestamp'].rank(method='first', ascending=False)
        candidate_train_test = ratings[ratings.duplicated(['account_id'], False)]
        candidate_train = ratings[~ratings.duplicated(['account_id'], False)]
        test = candidate_train_test[candidate_train_test ['rank_latest'] == 1].reset_index(drop=True)
        train = pd.concat([candidate_train, candidate_train_test[candidate_train_test['rank_latest'] > 1]], axis=0).reset_index(drop=True)
        print('training_size: %d, test_size: %d' % (len(train), len(test)))
        #assert train['account_id'].nunique() == test['account_id'].nunique()
        return train[['account_id', 'atlas_location_uuid', 'rating', 'weight']], test[['account_id', 'atlas_location_uuid', 'rating', 'weight']]

    # def _split_loo(self, ratings):
    #     """leave one out train/test split """
    #     train = ratings.sample(frac=0.9, random_state=0, axis=0)
    #     test = ratings[~ratings.index.isin(train.index)]
    #     return train, test

    def get_all_relevant_location_ids(self, location_ids):
        relevant_location_ids = set.union(*[self.item_pool['city_loc'][self.item_pool['loc_city'][x]] for x in location_ids])
        return relevant_location_ids
    
    def not_negative_location_ids(self, location_ids):
        covisit_df = self.covisit_df
        covisit_location = covisit_df[covisit_df['atlas_location_uuid'].isin(location_ids)]['atlas_location_uuid_covisit'].unique()

        covisit_location = set(covisit_location)
        return set.union(location_ids, covisit_location)

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('account_id')['atlas_location_uuid'].apply(set).reset_index().rename(
                            columns={'atlas_location_uuid': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.get_all_relevant_location_ids(x) - self.not_negative_location_ids(x))
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, min(99, len(x))))
        return interact_status[['account_id', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        
        train_ratings = pd.merge(self.train_ratings, self.negatives[['account_id', 'negative_items']], on='account_id')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, min(len(x), num_negatives)))
        train_negatives = train_ratings[['account_id', 'negatives']]

        for row in train_ratings.itertuples():
            users.append(int(row.account_id))
            items.append(int(row.atlas_location_uuid))
            ratings.append(float(row.rating))
            for i in range(min(num_negatives, len(row.negatives))):
                users.append(int(row.account_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), train_negatives

    def val(self):
        self.mode = 'val'
    def test(self):
        self.mode = 'test'
    @property
    def evaluate_data(self):
        """create evaluate data"""
        if self.mode == 'test':
            test_ratings = pd.merge(self.test_ratings, self.negatives[['account_id', 'negative_samples']], on='account_id')
        elif self.mode == 'val':
            test_ratings = pd.merge(self.val_ratings, self.negatives[['account_id', 'negative_samples']], on='account_id')
        test_users, test_items, test_weights, negative_users, negative_items,gold_scores = [], [], [], [], [], []
        weight_sum = test_ratings['weight'].sum()
        
        for row in test_ratings.itertuples():
            test_users.append(int(row.account_id))
            test_items.append(int(row.atlas_location_uuid))
            test_weights.append(float(row.weight) / weight_sum)
            gold_scores.append(row.rating)
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.account_id))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items), torch.FloatTensor(test_weights), torch.FloatTensor(gold_scores)]
