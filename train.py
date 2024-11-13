import time
import torch
import matplotlib.pyplot as plt
import copy
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(trainloader,valloader,model,optimizer,loss_fn,scheduler,num_epochs,patience,save_training_plot=False,training_data = 'IMDB'):
    time0 = time.time()
    callback_counter = 0
    best_val_loss = 100000
    best_epoch = 0
    train_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    val_loss_hist = []


    for epoch in range(num_epochs):
        train_batch_hist = []
        train_batch_loss = []
        model.train()
        for i, (data,targets) in enumerate(trainloader):
            if training_data == 'SHD':
                data = data.reshape(data.size(0),100,700).to(device)
                targets = targets.type(torch.LongTensor).to(device)
            else:
                data = data.to(device)
                targets = targets.to(device)
            output = model(data)
            loss_val = loss_fn(output,targets)

            #Backprop
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()
            train_batch_loss.append(loss_val.item())

            #print(f"Epoch {epoch}, Iteration {i}, \nTrain Loss: {loss_val.item():.2f}")
        
            pred_targets = torch.max(output,1).indices
            train_correct = torch.sum(torch.eq(pred_targets,targets))
            train_acc = train_correct/len(targets)

            train_batch_hist.append(train_acc)
            #print(f"Accuracy: {train_acc*100:-2f}%\n")
        
        average_train_acc = sum(train_batch_hist)/len(train_batch_hist)
        train_acc_hist.append(average_train_acc)
        average_train_loss = sum(train_batch_loss)/len(train_batch_loss)
        train_loss_hist.append(average_train_loss)

        model.eval()
        with torch.no_grad():
            val_batch_hist = []
            val_batch_loss = []

            for (val_data,val_targets) in valloader:
                if training_data == 'SHD':
                    val_data = val_data.reshape(val_data.size(0),100,700).to(device)
                    val_targets = val_targets.type(torch.LongTensor).to(device)
                else:
                    val_data = val_data.to(device)
                    val_targets = val_targets.to(device)
                val_output = model(val_data)
                validation_loss = loss_fn(val_output,val_targets)
                val_batch_loss.append(validation_loss.item())

                
                
                pred_val_targets = torch.max(val_output,1).indices
                train_correct = torch.sum(torch.eq(pred_val_targets,val_targets))
                val_acc = train_correct/len(val_targets)

                val_batch_hist.append(val_acc)

            average_val_acc = sum(val_batch_hist)/len(val_batch_hist)
            val_acc_hist.append(average_val_acc)
            average_val_loss = sum(val_batch_loss)/len(val_batch_loss)
            val_loss_hist.append(average_val_loss)


        print(f"End of epoch {epoch}:")
        print(f"Average train set accuracy: {average_train_acc*100:-2f}%")
        print(f"Average validation set accuracy: {average_val_acc*100:-2f}%")
        print(f"Average train set loss: {average_train_loss:-2f}")
        print(f"Average validation set loss: {average_val_loss:-2f}\n")

        #Callback to save best model for inference
        if average_val_loss >= best_val_loss:
            callback_counter +=1
        else:
            best_val_loss = average_val_loss
            callback_counter = 0
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
        if callback_counter >= patience:
            print(f'Training stopped at epoch {epoch} after {patience} iterations without improvement.')
            print(f'The best validation loss was: {best_val_loss}, at epoch {best_epoch}')
            break

    print("\n[INFO] Total Time (in minutes) =", (time.time() - time0) / 60)
    if save_training_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        new_train_acc_hist = []
        new_val_acc_hist = []
        for acc,valacc in zip(train_acc_hist,val_acc_hist):
            acc = acc.cpu()
            valacc = valacc.cpu()
            new_train_acc_hist.append(acc)
            new_val_acc_hist.append(valacc)
        plt.plot(new_train_acc_hist,label="Train Acc")
        plt.plot(new_val_acc_hist,label = "Val Acc")
        plt.title("Tr_acc vs Val_acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        ax1.plot(train_loss_hist,label= "Train loss")
        ax1.plot(val_loss_hist,label= "Validation loss")
        ax1.title.set_text("tr_loss vs Val_loss")
        ax1.legend()
        plt.savefig(str(os.path.abspath('') +'\\saved_figures\\'+ training_data +'.png'))
    return model,best_model_state_dict