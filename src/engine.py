import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import save_checkpoint, resume_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK()
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        if not config['implicit']:
        # explicit feedback
            self.crit = torch.nn.MSELoss()
        else:
        # implicit feedback
            self.crit = torch.nn.BCEWithLogitsLoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_logits = self.model(users, items)
        # print (ratings_pred)
        # print (ratings)
        loss = self.crit(ratings_logits.view(-1), ratings)
        loss.backward()
        
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, ratings = batch[0], batch[1], batch[2]
            ratings = ratings.float()
            loss = self.train_single_batch(users, items, ratings)
            
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, train_negatives,  epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            test_weights, gold_scores = evaluate_data[4], evaluate_data[5]
        
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            # print (test_scores.data.view(-1).tolist())
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist(),
                                 test_weights.data.view(-1).tolist(),
                                 gold_scores.data.view(-1).tolist()]

        print('[Evluating Epoch {}]'.format(epoch_id))
        basic_metric = 0
        if not self.config['implicit']:             
            mae = self._metron.cal_basic_metric(train_negatives, 'mae')
            print ('mae = {:.4f}'.format(mae))
            basic_metric = mae
        else:
            acc = self._metron.cal_basic_metric(train_negatives, 'acc')
            print ('acc = {:.4f}'.format(acc))
            basic_metric = acc
        auc = self._metron.cal_basic_metric(train_negatives, 'auc')
        print ('auc = {:.4f}'.format(auc))
        #hit_ratio1  = self._metron.cal_hit_ratio(1)
        hit_ratio5  = self._metron.cal_hit_ratio(5)
        hit_ratio10  = self._metron.cal_hit_ratio(10)
        #ndcg1 =  self._metron.cal_ndcg(1)
        ndcg5 = self._metron.cal_ndcg(5)
        ndcg10 = self._metron.cal_ndcg(10)
        print('HR@5 = {:.4f}, HR@10 = {:.4f},\
             NDCG@5 = {:.4f}, NDCG@10 = {:.4f}'
            .format(hit_ratio5, hit_ratio10, ndcg5, ndcg10))
        # self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        # self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        return basic_metric, auc, hit_ratio5, hit_ratio10, ndcg5, ndcg10

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)

    def load(self, alias, epoch_id, hit_ratio, ndcg):
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        device_id = 0
        if self.config['use_cuda'] is True:
            device_id = self.config['device_id']
        resume_checkpoint(self.model, model_dir=model_dir , device_id=device_id)
        return self.model
