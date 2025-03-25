import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
#from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
import math
#torch.manual_seed(1)
#np.random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

    
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states

class data_process_loader(data.Dataset):
	def __init__(self, list_IDs, labels, drug_df,rna_df):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.drug_df = drug_df
		self.rna_df = rna_df

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		#index = self.list_IDs[index]
		v_d = self.drug_df.iloc[index]['drug_encoding']
		v_p = np.array(self.rna_df.iloc[index])
		y = self.labels[index]

		return v_d, v_p, y

class transformer(nn.Sequential):
    def __init__(self):
        super(transformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]

class MLP(nn.Sequential):
    def __init__(self):
        input_dim_gene = 17417 #changed from 17737 n.b.
        hidden_dim_gene = 256
        mlp_hidden_dims_gene = [1024, 256, 64]
        super(MLP, self).__init__()
        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v


class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_gene, input_dim_drug=128):
        super(Classifier, self).__init__()
        self.input_dim_drug = input_dim_drug
        self.input_dim_gene = 256
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout = nn.Dropout(0.1)
        self.hidden_dims =  [1024, 1024, 512]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_gene] + self.hidden_dims + [1]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_gene(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f
    



#early stopping implementation
class EarlyStopping:
    '''Class to implement early stopping
    
     ------inputs------
    model: PyTorch model 
    
    patience: int, defalt=10
    number of epochs the model can not meet the improvement criteria before
    early stopping triggers
    
    delta: int, defalt=0
    How much the loss needs to decrease by to count as an improvement
    e.g. delta=1 means the loss needs to be at least 1 less than previous best loss
    '''
    def __init__(self, patience=10, delta=0.0, verb=0, modeldir=None):
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.min_val_loss = np.inf
        self.best_model_dict = None
        self.best_epoch = None
        self.verb = verb
        self.modeldir = modeldir
        
    def earily_stop(self, val_loss, model_state, e=0):
        #if loss improves 
        if val_loss < self.min_val_loss:
            if self.verb > 0:
                print(f'loss improved from {self.min_val_loss} to {val_loss}')
            self.min_val_loss = val_loss
            self.count = 0 
            #self.best_model_dict = model_state
            torch.save(model_state, f'{self.modeldir}tta')
        #if loss does not improved more than delta 
        elif val_loss >= (self.min_val_loss + self.delta):
            self.count += 1
            if self.count >= self.patience: 
                return True
            
        return False #if stopping contions not met
        
class Marker_drug(nn.Sequential):
    '''Just returns drug encoding'''
    def forward(self, xd):
        xd = xd.float().to(device)
        return xd
class DeepTTC:
    def __init__(self, modeldir, drug_branch=True):
        if drug_branch:
            model_drug = transformer()
            input_dim_drug = 128
        else:
            model_drug = Marker_drug()
            input_dim_drug = 181 #177 for binary nb
        model_gene = MLP()
        self.model = Classifier(model_drug, model_gene, 
                                input_dim_drug=input_dim_drug)
        self.device = torch.device('cuda:0')
        self.modeldir = modeldir
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = f'{self.modeldir}_loss_curve_iter.pkl'

    def test(self,datagenerator,model):
        y_label = []
        y_pred = []
        model.eval()
        for i,(v_drug,v_gene,label) in enumerate(datagenerator):
            score = model(v_drug,v_gene)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        model.train()

        return y_label, y_pred, \
               mean_squared_error(y_label, y_pred), \
               np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               pearsonr(y_label, y_pred)[1], \
               spearmanr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[1], \
               loss
               #concordance_index(y_label, y_pred), \

    def train(self, train_drug, train_rna, train_y,
              val_drug, val_rna, val_y, 
              patience=300, train_epoch=300):
        lr = 1e-4 
        decay = 0
        BATCH_SIZE = 128 #128 from paper
        self.model = self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 5])
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  #'num_workers': 0,
                  'drop_last': False}
        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_y, train_drug, train_rna), **params)
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_y, val_drug, val_rna), **params)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ['# epoch',"MSE", 'RMSE',
                                    "Pearson Correlation", "with p-value",
                                    'Spearman Correlation',"with p-value2",
                                    "Concordance Index"]
        #table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x
        print('--- Go for Training ---')
        #writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0

        if patience:
            verb=0
            early_stopper = EarlyStopping(patience=patience, delta=0, 
                                          modeldir=self.modeldir,
                                          verb=verb
                                          )


        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                # print(v_d,v_p)
                # v_d = v_d.float().to(self.device)
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                #writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                          ' iteration ' + str(i) + \
                          ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                          ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            with torch.set_grad_enabled(False):
                ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                y_true,y_pred, mse, rmse, \
                person, p_val, \
                spearman, s_p_val ,\
                loss_val = self.test(validation_generator, self.model)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val]))
                valid_metric_record.append(lst)
                if patience and early_stopper.earily_stop(
                    mse, self.model.state_dict()):
                        print('')
                        print(f'stopping early at epoch {epo + 1}')
                        print(f'best epoch {epo + 1 - early_stopper.patience}')
                        break


                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + str(loss_val.item())[:7] +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val)[:7] +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val)[:7]
                          #' , Concordance Index: ' + str(CI)[:7]
                          )
                    # writer.add_scalar("valid/mse", mse, epo)
                    # writer.add_scalar('valida/rmse', rmse, epo)
                    # writer.add_scalar("valid/pearson_correlation", person, epo)
                    #writer.add_scalar("valid/concordance_index", CI, epo)
                    # writer.add_scalar("valid/Spearman", spearman, epo)
                    # writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
            #table.add_row(lst)

        #self.model = model_max
        #load best model if early stoppng triggers or not
        if patience:
            print('loading best model')
            #best_model_dict  = early_stopper.best_model_dict
            #model.load_state_dict(best_model_dict)
            self.model.load_state_dict(torch.load(f'{self.modeldir}tta'))
            self.model.eval()

        #with open(self.record_file, 'w') as fp:
            #fp.write(table.get_string())
        loss_history = pd.DataFrame(loss_history)
        loss_history.to_csv(f'{self.modeldir}hist')

        print('--- Training Finished ---')
        #writer.flush()
        #writer.close()

    def predict(self, drug_data, rna_data, y):
        print('predicting...')
        self.model.to(device)
        info = data_process_loader(drug_data.index.values,
                                   y, drug_data, rna_data)
        # params = {'batch_size': 16,
        #           'shuffle': False,
        #           'num_workers': 8,
        #           'drop_last': False,
        #           'sampler': SequentialSampler(info)}
        generator = data.DataLoader(info) #**params)

        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, loss_val = \
            self.test(generator, self.model)

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val

    def save_model(self):
        torch.save(self.model.state_dict(), self.modeldir + '/model.pt')

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

