import keras
from torch.utils.data import TensorDataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
from imdb_model import TransformerModelIMDB
import torch.nn as nn
from train import train

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.utils.pad_sequences(x_test, maxlen=maxlen)

    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size=0.75,random_state=0)

    batch_size = 32

    # create tensor datasets
    trainset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    validset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    testset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    # create dataloaders
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    #create the model, optimizer, loss_fn and scheduler
    model = TransformerModelIMDB(embed_dim=32,num_heads=2,d_k=32,ff_dim=32,max_len=maxlen,vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3,betas = (0.9,0.999))
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    model, best_model_state_dict = train(trainloader=trainloader,valloader=valloader,model=model,
                                         optimizer=optimizer,scheduler=scheduler,loss_fn=loss_fn,
                                         num_epochs=10,patience=3,save_training_plot=True,training_data='IMDBtraining')

    test_acc_hist = []
    for i, (data,targets) in enumerate(testloader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            test_output = model(data)
            pred_targets = torch.max(test_output,1).indices
            train_correct = torch.sum(torch.eq(pred_targets,targets))
            acc = train_correct/len(targets)
            test_acc_hist.append(acc)
    print(f"The average accuracy across de testloader of the last model is {sum(test_acc_hist)/len(test_acc_hist)}")

    best_model = TransformerModelIMDB(embed_dim=32,num_heads=2,d_k=32,ff_dim=32,max_len=maxlen,vocab_size=vocab_size).to(device)
    best_model.load_state_dict(best_model_state_dict)

    best_test_acc_hist = []
    for i, (data,targets) in enumerate(testloader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            test_output = best_model(data)
            pred_targets = torch.max(test_output,1).indices
            train_correct = torch.sum(torch.eq(pred_targets,targets))
            acc = train_correct/len(targets)
            best_test_acc_hist.append(acc)
    print(f"The average accuracy across de testloader of the best model is {sum(best_test_acc_hist)/len(best_test_acc_hist)}")


if __name__ == '__main__':
    main()