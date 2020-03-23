import math
import pandas as pd
from sklearn.metrics import roc_auc_score,  mean_absolute_error, accuracy_score

class MetronAtK(object):
    def __init__(self):
        # self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        test_weights, gold_scores = subjects[6],subjects[7]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,   
                             'test_score': test_scores,
                             'weight': test_weights,
                             'gold_score': gold_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full
    
    def cal_basic_metric(self, train_negative, metric_type):
        full = self._subjects
        positive = full[full['test_item'] == full['item']]
        negative = full[full['test_item'] != full['item']]  
        test_negative = pd.merge(negative, train_negative, how='left', left_on=['user', 'item'], right_on=['account_id', 'negative_id'])
        test_negative = test_negative[~((test_negative.user == test_negative.account_id) & 
                                        (test_negative.item == test_negative.negative_id))]
        test_negative_sample = test_negative.sample(len(positive)*2)
        y_prob = positive['score'].tolist() + test_negative_sample['score'].tolist()
        y_pred = []
        # y_true = []
        for prob, gold in zip(y_prob, positive['gold_score'].tolist()):
            if prob >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
            # if gold > 0:
            #     y_true.append(1)
            # else:
            #     y_true.append(0)
        if metric_type == 'mae':
            y_true = positive['gold_score'].tolist() + [0]*len(test_negative_sample)
            return mean_absolute_error(y_true, y_prob)
        elif metric_type == 'auc':
            y_true = [1]*len(positive['gold_score']) + [0]*len(test_negative_sample)
            return roc_auc_score(y_true, y_prob)
        elif metric_type == 'acc':
            y_true = [1]*len(positive['gold_score']) #+ [0]*len(test_negative_sample)
            return accuracy_score(y_true, y_pred)

    # def cal_acc(self, train_negative):
    #     full = self._subjects
    #     positive = full[full['test_item'] == full['item']]
    #     negative = full[full['test_item'] != full['item']]  
    #     test_negative = pd.merge(negative, train_negative, how='left', left_on=['user', 'item'], right_on=['account_id', 'negative_id'])
    #     test_negative = test_negative[~((test_negative.user == test_negative.account_id) & 
    #                                     (test_negative.item == test_negative.negative_id))]
    #     test_negative = test_negative.sample(len(positive))
    #     y_true = positive['gold_score'].tolist() + [0]*len(test_negative)
    #     y_pred = positive['score'].tolist() + test_negative['score'].tolist()
    #     return mean_absolute_error(y_true, y_pred)

    # def cal_auc(self, train_negative):
    #     full = self._subjects
    #     positive = full[full['test_item'] == full['item']]
    #     negative = full[full['test_item'] != full['item']]
    #     test_negative = pd.merge(negative, train_negative, how='left',left_on=['user', 'item'], right_on=['account_id', 'negative_id'])
    #     # print (negative.sample(10))
    #     # print (train_negative.sample(10))
    #     test_negative = test_negative[~((test_negative.user == test_negative.account_id) & 
    #                                     (test_negative.item == test_negative.negative_id))]
    #     print (len(test_negative), len(positive))
    #     test_negative = test_negative.sample(len(positive))
    #     y_true = [1]*len(positive) + [0]*len(test_negative)
    #     y_pred = positive['score'].tolist() + test_negative['score'].tolist()
    #     return roc_auc_score(y_true, y_pred)

    def cal_hit_ratio(self, top_k):
        """Hit Ratio @ top_K"""
        full = self._subjects
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        #return len(test_in_top_k) * 1.0 / full['user'].nunique()
        return test_in_top_k['weight'].sum()

    def cal_ndcg(self, top_k):
        full = self._subjects
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        #print (test_in_top_k)
        #test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        #return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
        test_in_top_k['ndcg'] = test_in_top_k.apply(lambda x: x['weight'] * math.log(2) / math.log(1 + x['rank']), axis=1) # the rank starts from 1
        return test_in_top_k['ndcg'].sum()