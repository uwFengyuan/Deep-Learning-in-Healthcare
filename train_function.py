import torch
from torch.autograd import Variable
import numpy as np
import torchio as tio
import time
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def L2RegLoss(main_loss, x, target, mu, model):

    main_loss = main_loss(x, target)
    penalty = 0
    for i in model.parameters():
        penalty += torch.sum(i**2)
    J = main_loss + mu*penalty

    return J

def train(net, dataloader, optim, loss_func, loss_func1, epoch):
    net.train()  #Put the network in train mode
    total_loss = 0
    batches = 0
    
    start = time.time()

    for batch_idx, batch in enumerate(dataloader):

        #if hp.debug:
        #    if batch_idx >=1:
        #        break
        data = batch['image'][tio.DATA]
        target = batch['label'][tio.DATA]
        data, target = Variable(data).to(device), Variable(target).to(device)
        batches += 1

        # Training loop
        optim.zero_grad()
        pred = net(data) 
        target = target.to(torch.float32)
        if loss_func1 is None:
            loss = L2RegLoss(loss_func, pred, target, 0.0001,
                             net)  # loss_func(pred ,target)
        else:
            loss = L2RegLoss(loss_func, pred, target, 0.0001,
                             net) + L2RegLoss(loss_func1, pred, target, 0.0001,
                             net) #loss_func(pred, target) + loss_func1(pred, target)

        loss.backward()
        optim.step()
        
        total_loss += loss
        if batch_idx % 4 == 0: #Report stats every x batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(dataloader.dataset),
                        100. * (batch_idx+1) / len(dataloader), loss.item()), flush=True)
    av_loss = total_loss / batches
    av_loss = av_loss.detach().cpu().numpy()
    print('Training set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    total_time = time.time() - start
    print('Time taken for epoch = ', total_time)
    return av_loss

def val(net, val_dataloader, optim, loss_func, loss_func1):
    net.eval()  #Put the model in eval mode
    total_loss = 0    
    batches = 0
    with torch.no_grad():  # So no gradients accumulate
        for batch_idx, batch in enumerate(val_dataloader):
            batches += 1
            data = batch['image'][tio.DATA]
            target = batch['label'][tio.DATA]
            data, target = Variable(data).to(device), Variable(target).to(device)
            # Eval steps
            optim.zero_grad()
            target = target.to(torch.float32)

            pred = net(data) 
            if loss_func1 is None:
                loss = loss_func(pred ,target)
            else:
                loss = loss_func(pred, target) + loss_func1(pred, target)
            
            total_loss += loss
        av_loss = total_loss / batches
        
    av_loss = av_loss.detach().numpy()
    print('Validation set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('\n')
    return av_loss

def predict(net, test_dataloader):
    pred_store = []
    true_store = []
         
    with torch.no_grad():  # So no gradients accumulate
        for batch_idx, batch in enumerate(test_dataloader):
            data = batch['image'][tio.DATA]
            target = batch['label'][tio.DATA]
            data, target = Variable(data).to(device), Variable(target).to(device)
            # Complete validation loop here:
            target = target.to(torch.float32)
            pred = net(data) 
            #print(pred.shape)
            #print(np.argmax(pred.cpu().detach().numpy(), axis=1).shape)
            pred_store.extend(np.argmax(pred.cpu().detach().numpy(), axis=1))

            true_store.extend(np.argmax(target.cpu().detach().numpy(), axis=1))
    
    pred_store = torch.from_numpy(np.array(pred_store)).clone()
    pred_store = torch.reshape(pred_store.detach(), (pred_store.shape[0], 1, pred_store.shape[1], \
                                                            pred_store.shape[2], pred_store.shape[3]))
    pred_store = np.array(pred_store)

    true_store = torch.from_numpy(np.array(true_store)).clone()
    true_store = torch.reshape(true_store.detach(), (true_store.shape[0], 1, true_store.shape[1], \
                                                            true_store.shape[2], true_store.shape[3]))
    true_store = np.array(true_store)
    

    return pred_store, true_store

def count_parameters(model: torch.nn.Module) -> int: # return type is int
    """ Returns the number of learnable parameters for a PyTorch model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)