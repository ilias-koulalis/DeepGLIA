import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models import CNN_LSTM, CNN, CNN_LSTM2
import pandas as pd
from datetime import datetime
from copy import deepcopy

seed = torch.Generator().manual_seed(17912)

'''
Specifying run parameters
'''
MODEL = CNN_LSTM2
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
AUGMENTED_DATASET = False
SAVE_PREDICTIONS = True

'''
train function
'''
def train_model(train, val, model, epochs=40, batch_size=128, learning_rate=1e-3, save_predictions=False, rc_augmented=False):

    #Loading dataset into batched input
    valloader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle=False, num_workers=1)
    trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=False, num_workers=1)   

    #Setting device (cpu/gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model()
    model.to(device)

    #Specifiying the loss function and optimizer
    criterion = nn.BCELoss().cuda(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    #Initialize log and log hyperparams
    log_dir = f"../runs/{model.name}_{datetime.now().strftime('%d-%m_%H:%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    with open(f'{log_dir}/hyperparams.log', "w") as hyperparams_log:
        hyperparams_log.write(f'Model: {model}\n'+\
                        f'{pd.DataFrame(model.get_parameters(),columns=("LAYER", "PARAMS"))}\n'+\
                        f'epochs:{epochs}\tbatch_size:{batch_size}\tlearning_rate:{learning_rate}\n'+\
                        f'train size: {len(train)}\tval size: {len(val)}\n'+\
                        f'Augmented dataset: {rc_augmented}\n'+\
                        f'Using:{device}\n')

    #Train loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        
        #Training
        model.train()

        running_loss = 0.0
        for i, data in enumerate(trainloader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, targets]
            # store the tensors in the device memory
            inputs, targets = data[0].to(device), data[1].to(device)
            
            if model.name=='CNN_LSTM2':
                inputs_rc = np.array(data[0])[:,::-1,::-1]
                inputs_rc = torch.Tensor(inputs_rc.copy()).to(device)
                outputs = model(inputs, inputs_rc)
                
            else:
                outputs = model(inputs)

            # forward + backward prop / optimize
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            
        average_loss = running_loss / (i+1)

        #Validation
        model.eval()

        running_vloss = 0.0
        for i, vdata in enumerate(valloader):
            vinputs, vtargets = vdata[0].to(device), vdata[1].to(device)

            if model.name=='CNN_LSTM2':
                vinputs_rc = np.array(vdata[0])[:,::-1,::-1]
                vinputs_rc = torch.Tensor(vinputs_rc.copy()).to(device)
                voutputs = model(vinputs, vinputs_rc)

            else:
                voutputs = model(vinputs)

            validation_loss = criterion(voutputs, vtargets)
            running_vloss += validation_loss.item() 

        average_vloss = running_vloss / (i+1)

        writer.add_scalar('train loss/epoch', average_loss, epoch+1)
        writer.add_scalar('validation loss/epoch', average_vloss, epoch+1)

        torch.cuda.empty_cache()

        #Save best model
        if average_vloss < best_val_loss:
            best_val_loss = average_vloss
            best_epoch = epoch
            state_dict = deepcopy(model.state_dict())

    with open(f'{log_dir}/hyperparams.log', "a") as hyperparams_log:
        hyperparams_log.write(f'Lowest validation loss at epoch {best_epoch}')

    #Saving final parameters in a file
    PATH = f'./{writer.get_logdir()}/state_dict.pth'
    torch.save(state_dict, PATH)
    writer.flush()
    writer.close()
    print(f'Finished Training\nWritten to: {PATH}')

    #Save predictions for finalized params
    if save_predictions:
        model.eval()
        model.load_state_dict(state_dict)     

        predictions = np.zeros((0,len(targets.T)))
        labels = np.zeros((0,len(targets.T)))
        vpredictions = np.zeros((0,len(targets.T)))
        vlabels = np.zeros((0,len(targets.T)))
        with torch.no_grad():
            for data in trainloader:
                inputs, targets = data[0].to(device), data[1]

                if model.name=='CNN_LSTM2':
                    inputs_rc = np.array(inputs)[:,::-1,::-1]
                    inputs_rc = torch.Tensor(inputs_rc.copy()).to(device)
                    outputs = model(inputs, inputs_rc)
                else:
                    outputs = model(inputs)

                predictions = np.append(predictions,outputs.detach().cpu(),axis=0)
                labels = np.append(labels, targets.detach(),axis=0)

            for vdata in valloader:
                vinputs, vtargets = vdata[0].to(device), vdata[1]

                if model.name=='CNN_LSTM2':
                    vinputs_rc = np.array(vinputs)[:,::-1,::-1]
                    vinputs_rc = torch.Tensor(vinputs_rc.copy()).to(device)
                    voutputs = model(vinputs, vinputs_rc)
                else:
                    voutputs = model(vinputs)

                vpredictions = np.append(vpredictions,voutputs.detach().cpu(),axis=0)
                vlabels = np.append(vlabels, vtargets.detach(),axis=0)

        prediction_dict = {"train": {}, "val": {}}
        prediction_dict["train"]["prediction"] = predictions
        prediction_dict["train"]["labels"] = labels
        prediction_dict["val"]["prediction"] = vpredictions
        prediction_dict["val"]["labels"] = vlabels


        with open(f'{log_dir}/predictions.pickle', 'wb') as file:
            pickle.dump(prediction_dict, file)

          
if __name__ == "__main__":

    np_data = np.load('../dataset/train_data.npy')
    np_targets = np.load('../dataset/train_targets.npy')

    #data into tensors
    data = torch.Tensor(np_data).transpose(1,2)      #for nn.Conv1D input must have shape (batch,in_channels=4,in_sequence=500)
    targets = torch.Tensor(np_targets)

    #merging into pytorch dataset
    dataset = torch.utils.data.TensorDataset(data, targets)

    #splitting into test and val
    train_size = int(0.8889 * len(dataset))
    train, val = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator=seed)

    # Augmentation
    #casting into new np.array to reverse along axes 1&2, since torch.Tensor does not allow indexing with step of 2
    np_rev_comp = np_data[train.indices][:,::-1,::-1].copy()
    np_rev_comp_targets = np_targets[train.indices].copy()
    #rev_comp data and targets into dataset
    train_rev_comp = torch.utils.data.TensorDataset(torch.Tensor(np_rev_comp).transpose(1,2), torch.Tensor(np_rev_comp_targets))
    train_augmented=torch.utils.data.ConcatDataset([train, train_rev_comp])


    TRAIN_SET = train
    if AUGMENTED_DATASET and MODEL!=CNN_LSTM2:
        TRAIN_SET = train_augmented

    train_model(train=TRAIN_SET, val=val, model=MODEL,epochs=EPOCHS,batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                 save_predictions=SAVE_PREDICTIONS, rc_augmented=AUGMENTED_DATASET)