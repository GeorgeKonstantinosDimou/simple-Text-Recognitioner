import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import Utils
from Dataset import MyDataset
import Model


def train_model(model, params, optimizer, lr_scheduler, dataloader):
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    utilities = Utils.Utilities()
    
    for epoch in range(params['epochs']):
        
        print(f"Epoch {epoch}/{params['epochs'] -1}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            running_totals = 0
            
            for imgs, targets, length in dataloader[phase]:
                imgs = imgs.to(params['device'])
                targets = targets.to(params['device'])
                
                #print(imgs.shape)
                #print(length)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs, targets)
                    #print(outputs.shape)
                    #print(targets)
                    
                    outputs_length = torch.IntTensor([outputs.size(0)] * imgs.size(0))
                    log_prob = outputs.log_softmax(2)
                    outputs = outputs.permute(1, 0, 2)
                    _, preds = torch.max(outputs, 2)
                    #print(preds)
                    #print(preds.shape)
                    
                    losses = params['loss_fn'](log_prob, targets, outputs_length, length)
                    #print(losses)
                    
                    if phase == 'train': 
                        losses.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 3.0)
                        optimizer.step()
                  
                running_loss += losses.item()
                
                preds = preds.contiguous().view(-1)
                condition = (preds != 0) & (preds != 38)
                preds_non_unec = preds[condition]
                predicted_labels = utilities.remove_unneces(preds_non_unec, torch.sum(length))
                
                targets_flattened = targets.view(-1)
                targets_non_padded = targets_flattened[targets_flattened != 38]
                # if predicted_labels.size() == targets_non_padded.size():
                #     print("YES")
                #     print(f"Predicted and targets are both{predicted_labels.size()} {targets_non_padded.size()}")
                # else:
                #     print('NO')
                #     print(f"Predicted_labels is equal {predicted_labels.size()}")
                #     print(f"targets is equal {targets_non_padded.size()}")

                running_corrects += sum([preds == trg for preds, trg in zip(predicted_labels, targets_non_padded)])
                #running_corrects += torch.sum(torch.eq(predicted_labels, targets_non_padded))
                # for preds, trg in zip(predicted_labels, targets_non_padded):
                #     running_corrects += sum(preds == trg)
                #     print(f"preds are: {preds}")
                #     print(f"targets are: {trg}")
                running_totals += targets_non_padded.size(dim=0)
                
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / dataloader[phase].__len__()
            print(f"Running correctes are: {running_corrects}") ###
            print(f"Running totals are: {running_totals}")
            epoch_acc = (running_corrects / running_totals)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ") 
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    
    model.load_state_dict(best_model_wts)
    return model

def main():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    
    params = {'num_layers': 1,
              'feature_layers' : 256,
              'hidden_dim' : 256,
              'vocab_size' : 36,
              'out_seq_len' : 30,
              #'embedding_size' : 2048,
              'batch_size' : 16,
              'device' : device,
              'lr' : 0.1,
              'weight_decay' : 0.00001,
              'epochs': 40,
              'loss_fn' : nn.CTCLoss(zero_infinity = True)
              }
    
    transform = transforms.Compose([
        transforms.Resize((48, 160)),  # (h, w)   48, 160   6 40
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)), #Initial
        transforms.Normalize((0.5261, 0.5018, 0.4856),
                             (0.2224, 0.2196, 0.1959)), #37, 38
        ])
        
    trainset = MyDataset(
        'IIIT5K_train', transform)
    
    testset = MyDataset(
        'IIIT5K_test', transform)

    sets = {'train': trainset,
            'val': testset}

    image_dataset = {x: sets[x] 
                     for x in ['train', 'val']}
    
    """This 4 lines are responsible for the calculation of mean and std 
        with of course the help of the get_mean_std method
        DONT FORGET: To comment out the .Normalize function"""
    # loader = torch.utils.data.DataLoader(image_dataset['train'], batch_size=len(image_dataset['train']))
    # m, s = 0, 0
    # m, s = trainset.get_mean_std()
    # print(m, s)
    
    dataloader = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=params['batch_size'], num_workers = 4, drop_last = True) 
                    for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_dataset[x]) 
                     for x in ['train', 'val']}

    # model = Model.ScratchModel(params)
    # model = model.to(params['device'])
    
    # optimizer = torch.optim.SGD(model.parameters(), params['lr'], momentum = 0.9, weight_decay = params['weight_decay'])
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.5, total_iters = 5)
    
    # model_ft = train_model(model, params, optimizer, lr_scheduler, dataloader)
    
    modelPre = Model.ModelPreTrained(params)
    modelPre = modelPre.to(params['device'])
 
    optimizerPre = torch.optim.SGD(modelPre.parameters(), params['lr'], momentum = 0.9, weight_decay = params['weight_decay'])
    lr_schedulerPre = torch.optim.lr_scheduler.LinearLR(optimizerPre, start_factor = 0.5, total_iters = 5)    
    
    model_ft = train_model(modelPre, params, optimizerPre, lr_schedulerPre, dataloader)

if __name__ == '__main__':
    main()