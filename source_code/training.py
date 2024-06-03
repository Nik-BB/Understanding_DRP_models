'''Pytorch traning loops

'''
import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, patience=10, delta=0.0, verb=0, ms_path=''):
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.min_val_loss = np.inf
        self.best_model_dict = None
        self.best_epoch = None
        self.verb = verb
        self.ms_path = ms_path
        
    def earily_stop(self, val_loss, model_state, e=0):
        #if loss improves 
        if val_loss < self.min_val_loss:
            if self.verb > 0:
                print(f'loss improved from {self.min_val_loss} to {val_loss}')
            self.min_val_loss = val_loss
            self.count = 0 
            #self.best_model_dict = model_state
            #Save
            torch.save(model_state, self.ms_path)
        #if loss does not improved more than delta 
        elif val_loss >= (self.min_val_loss + self.delta):
            self.count += 1
            if self.count >= self.patience: 
                return True
            
        return False #if stopping contions not met
        


#train loop for mutiple dataloaders 
def tl_multi_dls(train_dls=None, y_train=None, val_dls=None, y_val=None, 
                 model=None, loss_fn=nn.MSELoss(), optimiser=None, 
                 epochs=10, verb=1, early_stopping_dict=None,
                 ms_path=''):
    '''torch train loop for mutiple data loaders
    
    ------inputs------
    train_dls: list like 
    contains multiple inputs where each input is a data loader 
    for the traning data
    
    y_train: DataLoader
    dataloader with the target traning values
    
    val_dls: list like
    contains multiple inputs where each input is a data loader
    for the validaiton data
    
    y_val: DataLoader
    dataloader with the target valdiation values
    
    
    early_stopping_dict: dict
    contains earily stopping params, patience and delta 
    defalt patience=10 and delta=0.0
    
    ------returns------
    train_hist, best_model_dict
    
    if early stopping implemented best model is used to overwrite model
    this happens even when early stopping is't tirggered
    '''
    train_hist = {'train_loss': [], 'val_loss': []}
    best_model_dict = None
    assert type(train_dls) == list
    model = model.to(device)
    
    #early stopping
    if early_stopping_dict and val_dls:
        p, d = early_stopping_dict['patience'], early_stopping_dict['delta']
        early_stopper = EarlyStopping(patience=p, delta=d, verb=verb, 
                                      ms_path=ms_path)
    
    for e in range(epochs):
        #train_dls_y = train_dls
        #train_dls_y.append(y_train)
        loss_train = 0.0
        model.train()
        for batch, (*x, y) in enumerate(zip(*train_dls, y_train)):
            x = [inp.to(device) for inp in x]
            y = y.to(device)

            # Compute prediction error
            pred = model(*x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loss_train += loss.item()
            
        train_hist['train_loss'].append(loss_train / len(train_dls[0]))
        
        if val_dls:
            #val_dls_y = val_dls
            #val_dls_y.append(y_val)
            #find validaiton loss
            val_loss = 0.0
            r2 = 0.0
            mse_val = 0.0 

            model.eval()   
            with torch.no_grad():
                for *xv, yv in zip(*val_dls, y_val):
                    xv = [inp.to(device) for inp in xv]
                    yv = yv.to(device)
                    val_pred = model(*xv)
                    #val_pred = val_pred.reshape(len(val_pred))
                    v_loss = loss_fn(val_pred, yv)
                    val_loss += v_loss.item()
                    yv = yv.cpu().detach().numpy()
                    val_pred = val_pred.cpu().detach().numpy()
            train_hist['val_loss'].append(val_loss / len(val_dls[0]))
            
            #eairly stopping
            if early_stopping_dict:
                if early_stopper.earily_stop(val_loss, model.state_dict()):
                    print('')
                    print(f'stopping early at epoch {e + 1}')
                    print(f'best epoch {e + 1 - early_stopper.patience}')
                    #best_model_dict  = early_stopper.best_model_dict
                    #model.load_state_dict(best_model_dict)
                    #model.eval()
                    break
                    
        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dls[0])
            if val_dls:
                fvl = val_loss / len(val_dls[0])
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
                print(pearsonr(yv.reshape(len(yv)), 
                               val_pred.reshape(len(val_pred))))
            else:
                print(f'Train loss: {ftl:>5f}')
                
     #load best model if early stoppng triggers or not
    if early_stopping_dict:
        print('loading best model')
        #best_model_dict  = early_stopper.best_model_dict
        #model.load_state_dict(best_model_dict)
        model.load_state_dict(torch.load(ms_path))
        model.eval()
        
    return train_hist


#traning for py torch g graphs.
def tl_dual_graph(train_dl1=None, train_dl2=None, val_dl1=None, val_dl2=None, model=None, loss_fn=nn.MSELoss(), 
               optimiser=None, epochs=10, verb=1, early_stopping_dict=None, ms_path=''):
    '''torch train loop for two data loaders
    where train_dl1.y gives target values
    
    '''
    train_hist = {'train_loss': [], 'val_loss': []}
    model = model.to(device)
    
    #early stopping
    if early_stopping_dict and val_dl1:
        p, d = early_stopping_dict['patience'], early_stopping_dict['delta']
        early_stopper = EarlyStopping(patience=p, delta=d, ms_path=ms_path)
    
    for e in range(epochs):
        
        loss_train = 0.0
        model.train()
        for batch, (x1, x2) in enumerate(zip(train_dl1, train_dl2)):
            x1, x2 = x1.to(device), x2.to(device)
            y = x1.y.to(device)

            # Compute prediction error
            pred = model(x1, x2)
            #print(pred.shape)
            pred = pred.reshape(len(pred))
            loss = loss_fn(pred, y)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loss_train += loss.item()
            
        train_hist['train_loss'].append(loss_train / len(train_dl1))
        
        if val_dl1:
            #find validaiton loss
            val_loss = 0.0
            r2 = 0.0
            mse_val = 0.0 

            model.eval()   
            with torch.no_grad():
                for x1v, x2v in zip(val_dl1, val_dl2):
                    x1v, x2v = x1v.to(device), x2v.to(device) 
                    yv = x1v.y.to(device)
                    val_pred = model(x1v, x2v)
                    val_pred = val_pred.reshape(len(val_pred))
                    v_loss = loss_fn(val_pred, yv)
                    val_loss += v_loss.item()
                    yv = yv.cpu().detach().numpy()
                    val_pred = val_pred.cpu().detach().numpy()
                    #r2 = r2_score(yv, val_pred)
                    #mse_val = mean_squared_error(yv, val_pred)
                    #print(r2, mse_val)
            train_hist['val_loss'].append(val_loss / len(val_dl1))
            
            #eairly stopping
            if early_stopping_dict:
                if early_stopper.earily_stop(val_loss, model.state_dict()):
                    print('')
                    print(f'stopping early at epoch {e + 1}')
                    print(f'best epoch {e + 1 - early_stopper.patience}')
                    #bm_dict  = early_stopper.best_model_dict
                    #model.load_state_dict(bm_dict)
                    #model.eval()
                    break

        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dl1)
            if val_dl1:
                fvl = val_loss / len(val_dl1)
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
                print(pearsonr(yv.reshape(len(yv)), 
                               val_pred.reshape(len(val_pred))))
            else:
                print(f'Train loss: {ftl:>5f}')
                
     #load best model if early stoppng triggers or not
    if early_stopping_dict:
        model.load_state_dict(torch.load(ms_path))
        model.eval()
        
    return train_hist


