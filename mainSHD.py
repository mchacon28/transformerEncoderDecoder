import torch
from SHD_model import TransformerModelSHD
import torch.nn as nn
from train import train
import tonic
import os

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size=32
    seed  = 0
    train_split = 0.75
    val_split = 1-train_split
    sensor_size = tonic.datasets.SHD.sensor_size
    path = str((os.path.abspath('') + '\\datasets'))
    frame_transform = tonic.transforms.ToFrame( sensor_size=sensor_size,n_time_bins = 100)

    trainset = tonic.datasets.SHD(save_to = path , transform = frame_transform, train = True)
    testset = tonic.datasets.SHD(save_to = path , transform = frame_transform, train = False)


    cache_path_train = str(os.path.abspath('') + '\\cachedDatasets\\SHD\\train')
    cache_path_test = str(os.path.abspath('') + '\\cachedDatasets\\SHD\\test')
    cached_trainset = tonic.DiskCachedDataset(trainset,transform = torch.from_numpy, cache_path = cache_path_train)
    cached_testset = tonic.DiskCachedDataset(testset, transform = torch.from_numpy, cache_path = cache_path_test)
    (cached_trainset,cached_valset) = torch.utils.data.random_split(cached_trainset,[train_split,val_split],
                                                                generator = torch.Generator().manual_seed(seed))

    trainloader = torch.utils.data.DataLoader(cached_trainset, batch_size = batch_size, 
                                        collate_fn = tonic.collation.PadTensors(batch_first = True), shuffle = True)
    valloader = torch.utils.data.DataLoader(cached_valset, batch_size = batch_size, 
                                        collate_fn = tonic.collation.PadTensors(batch_first = True), shuffle = True)
    testloader = torch.utils.data.DataLoader(cached_testset, batch_size = batch_size, 
                                        collate_fn = tonic.collation.PadTensors(batch_first = True))

    model = TransformerModelSHD(embed_dim=700,num_heads=2,d_k=700,ff_dim=700,max_len=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3,betas = (0.9,0.999))
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    model, best_model_state_dict = train(trainloader=trainloader,valloader=valloader,model=model,
                                            optimizer=optimizer,scheduler=scheduler,loss_fn=loss_fn,
                                            num_epochs=200,patience=10,save_training_plot=True,training_data='SHD')

    test_acc_hist = []
    for i, (data,targets) in enumerate(testloader):
        data = data.reshape(data.size(0),100,700).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        with torch.no_grad():
            test_output = model(data)
            pred_targets = torch.max(test_output,1).indices
            train_correct = torch.sum(torch.eq(pred_targets,targets))
            acc = train_correct/len(targets)
            test_acc_hist.append(acc)
    print(f"The average accuracy across de testloader of the last model is {sum(test_acc_hist)/len(test_acc_hist)}")

    best_model = TransformerModelSHD(embed_dim=700,num_heads=2,d_k=700,ff_dim=700,max_len=100).to(device)
    best_model.load_state_dict(best_model_state_dict)

    best_test_acc_hist = []
    for i, (data,targets) in enumerate(testloader):
        data = data.reshape(data.size(0),100,700).to(device)
        targets = targets.type(torch.LongTensor).to(device)

        with torch.no_grad():
            test_output = best_model(data)
            pred_targets = torch.max(test_output,1).indices
            train_correct = torch.sum(torch.eq(pred_targets,targets))
            acc = train_correct/len(targets)
            best_test_acc_hist.append(acc)
    print(f"The average accuracy across de testloader of the best model is {sum(best_test_acc_hist)/len(best_test_acc_hist)}")


if __name__ == '__main__':
    main()