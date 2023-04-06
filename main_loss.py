import numpy as np
import monai
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import sys
from hparam import hparams as hp
from data_function import MedData
from train_function import train, val, predict, count_parameters
warnings.filterwarnings(action='ignore', category=FutureWarning)

def loss():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout = open('D_loss.log', mode = 'w',encoding='utf-8')
    print("Do the loss")

    aug = hp.aug[1]
    pre = hp.pre[1]
    steps = hp.steps[1]
    aug_command = hp.aug_commands[6]
    print('Batch_size: ', hp.batch_size)
    print('Total_epochs: ', hp.total_epochs)
    print('Debug: ', hp.debug)
    print('Augmenttion: ', aug)
    print('Preprocess: ', pre)
    print('Resolution Steps: ', steps)

    #for file in hp.files:
    file = 'ALL'
    print('File Source: ', file)
    print('Augmentation Type: ', aug_command)
    dataset = MedData(hp.image_dir_IOP, hp.label_dir_IOP, hp.image_dir_Guys, hp.label_dir_Guys,
                      hp.image_dir_HH, hp.label_dir_HH, hp.IOP_probability, hp.batch_size, aug_command)
    dataset.prepare_data()
    dataset.setup(aug, pre)
    
    print('Training:  ', len(dataset.train_set))
    print('Validation: ', len(dataset.val_set))
    print('Test:      ', len(dataset.test_set))

    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    from three_d.unet3d import UNet3D
    net = UNet3D(in_channels=1, out_channels=2,
                    init_features=hp.init_features, steps=steps).to(device)

    # Calculate the number of traininable params
    print('Trainable params: ', count_parameters(net))

    for this_class in hp.loss_functions:
        if this_class == 'DiceCE_Loss':
            class_loss = monai.losses.DiceCELoss(reduction='mean')
        elif this_class == 'Dice_Loss':
            class_loss = monai.losses.DiceLoss(reduction='mean')
        elif this_class == 'CrossE_Loss':
            class_loss = nn.CrossEntropyLoss()
        elif this_class == 'BCE_Loss':
            class_loss = nn.BCELoss()
        elif this_class == 'Focal_Loss':
            class_loss = monai.losses.FocalLoss()
        elif this_class == 'Tversky_Loss':
            class_loss = monai.losses.TverskyLoss()
        elif this_class == 'DiceFocal_Loss':
            class_loss = monai.losses.DiceFocalLoss()
        elif this_class == 'FocalTversky_Loss':
            class_loss = monai.losses.FocalLoss()
            class_loss1 = monai.losses.TverskyLoss()

        print("######   Loss Function is ######: ", this_class)

        optim = torch.optim.Adam(net.parameters(), lr=hp.init_lr)

        losses = []
        max_epochs = hp.total_epochs
        check_point = np.inf
        early_stop = 0
        stopping_pint = max_epochs
        if this_class == 'FocalTversky_Loss':
            for epoch in range(1, max_epochs+1):
                train_loss = train(net, train_dataloader, optim, class_loss, class_loss1, epoch)
                val_loss = val(net, val_dataloader, optim, class_loss, class_loss1)
                losses.append([train_loss, val_loss])
                if val_loss < check_point:
                    check_point = val_loss
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= hp.early_stop:
                    stopping_pint = epoch
                    print('Early stopping triggered. The final epoch is: ', epoch)
                    break
        else:
            for epoch in range(1, max_epochs+1):
                train_loss = train(net, train_dataloader, optim, class_loss, None, epoch)
                val_loss = val(net, val_dataloader, optim, class_loss, None)
                losses.append([train_loss, val_loss])
                if val_loss < check_point:
                    check_point = val_loss
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= hp.early_stop:
                    stopping_pint = epoch
                    print('Early stopping triggered. The final epoch is: ', epoch)
                    break

        losses = np.array(losses).T
        print('Losses Information (train, val): ', losses)
        its = np.linspace(1, stopping_pint, stopping_pint)

        plt.figure()
        plt.plot(its, losses[0, :])
        plt.plot(its, losses[1, :])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.savefig(f"{file}_{pre}_{aug}_{aug_command}_{class_loss}.png")
        plt.show()
        plt.close()

        pred, true = predict(net, test_dataloader)

        dice_metric = monai.metrics.DiceMetric(
            include_background=True, reduction="mean_batch")
        dice_score = dice_metric(y_pred=torch.from_numpy(
            pred), y=torch.from_numpy(true))
        print('Dice Value: ', torch.mean(dice_score).item())

        IoU_metric = monai.metrics.MeanIoU(
            include_background=True, reduction="mean_batch")
        IoU_socre = IoU_metric(y_pred=torch.from_numpy(
            pred), y=torch.from_numpy(true))
        print('IoU Score: ', torch.mean(IoU_socre).item())

loss()