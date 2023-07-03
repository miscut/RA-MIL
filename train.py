import csv
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import auc,roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import label_separate, read_bag, aaMILDataset
from model import RA_MIL
from utils import Regularization

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # gpu or cpu
    csv_path = 'label/fold_all.csv' # train label path
    data_path = 'dataset/t2_tumor' # train dataset path

    means = [0.15708263, 0.15708263, 0.15708263] # t2
    stds = [0.13905464, 0.13905464, 0.13905464]

    train_text = open("./evaluation/train.txt", "a")
    x = 1000  # modality num, T2:1000
    input_size = 224
    batch_size = 1
    epochs = 1000
    patience = 40
    lr = 1e-5
    wd = 1e-4
    nw = 16
    rs = 2

    print("using {} device.".format(device))
    print("using {} device.".format(device), file=train_text)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.ToTensor(),  # W，H，C to C，H，W, [0-1]
                                     transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))
                                     ]),
        "aug": transforms.Compose([transforms.Resize((input_size, input_size)),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5),
                                   transforms.RandomRotation(40),
                                   transforms.ToTensor(),
                                   transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))
                                   ]),
        "valid": transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))
                                     ])}

    csvfile = open(csv_path, encoding='UTF-8-sig')
    set_csv = csv.reader(csvfile)
    set_list = []
    for i, rows in enumerate(set_csv):
        set_list.append(rows)

    for s in set_list:
        # internal,5-fold
        # test_list = s  # test set is the specific set
        # training_list = [i for i in list(chain.from_iterable(set_list)) if i not in test_list]  # all - test

        # external,train&test
        training_list = s # if train with 241

        print("Fold {}.".format(str(set_list.index(s))), file=train_text)
        list_0, list_1 = label_separate(training_list)

        train_list_0, valid_list_0 = train_test_split(list_0, test_size=0.1, random_state=rs)
        train_list_1, valid_list_1 = train_test_split(list_1, test_size=0.1, random_state=rs)

        train_list = train_list_0 + train_list_1
        valid_list = valid_list_0 + valid_list_1

        random.shuffle(train_list)
        random.shuffle(valid_list)
        weight_0 = len(train_list_1) / len(train_list)
        weight_1 = len(train_list_0) / len(train_list)

        train_patient_list, train_label_list = read_bag(train_list, x, data_path)
        valid_patient_list, valid_label_list = read_bag(valid_list, x, data_path) # internal

        train_df = pd.DataFrame({'bag': train_patient_list, 'label': train_label_list})
        valid_df = pd.DataFrame({'bag': valid_patient_list, 'label': valid_label_list})

        train_dataset = aaMILDataset(train_df,transform=data_transform['aug'])
        valid_dataset = aaMILDataset(valid_df, transform=data_transform['valid'])
        train_num = len(train_dataset)
        val_num = len(valid_dataset)

        print('Using {} dataloader workers every process'.format(nw))
        print('Using {} dataloader workers every process'.format(nw), file=train_text)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        print("using {} bags for training, {} bags for validation.".format(train_num, val_num))
        print("using {} bags for training, {} bags for validation.".format(train_num, val_num), file=train_text)

        net = RA_MIL()
        net = net.to(device)
        if wd > 0:
            reg_loss = Regularization(net, wd, p=1).to(device)
        else:
            print("no regularization")

        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5,min_lr=1e-8)

        # train and validation
        best_acc = 0.0
        best_loss = 100
        best_auc = 0.0
        train_steps = len(train_loader)
        history = {'train_loss': [], 'val_loss': [], 'train_acc':[], 'val_acc': []}
        for epoch in range(epochs):

            # train
            print('lr = ', optimizer.state_dict()['param_groups'][0]['lr'])
            net.train()
            acc_t = 0.0
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            trn_0 = 0
            trn_1 = 0
            val_0 = 0
            val_1 = 0
            trn_rt_0 = 0
            trn_rt_1 = 0
            val_rt_0 = 0
            val_rt_1 = 0

            for step, data in enumerate(train_bar):
                _, bags, labels = data
                bags, labels = Variable(bags), Variable(labels)

                if labels == 0:
                    trn_0 += 1
                if labels == 1:
                    trn_1 += 1

                optimizer.zero_grad()
                bag_prob, predict_t, ins_prob,rank_ins = net(bags.to(device))

                # weighted loss
                weight = torch.zeros_like(labels).float().to(device)
                weight = torch.fill_(weight, weight_0)
                weight[labels == 1] = weight_1
                loss = nn.BCELoss(weight=weight)(bag_prob[0], labels.to(device).to(torch.float32))

                loss = loss + reg_loss(net)

                running_loss += loss.item()
                acc_t += torch.eq(predict_t, labels.to(device)).sum().item()
                loss.backward()
                optimizer.step()

                if predict_t == labels.to(device):
                    if labels == 1:
                        trn_rt_1 += 1
                    if labels == 0:
                        trn_rt_0 += 1
            print('train right prediction 0:', trn_rt_0, '/', trn_0)
            print('train right prediction 1:', trn_rt_1, '/', trn_1)
            print('train right prediction 0:', trn_rt_0, '/', trn_0, file=train_text)
            print('train right prediction 1:', trn_rt_1, '/', trn_1, file=train_text)
            train_loss = running_loss / len(train_loader)
            train_accurate = acc_t / len(train_loader.dataset)

            # validate
            net.eval()
            acc_val = 0.0
            running_valid_loss = 0
            with torch.no_grad():
                val_bar = tqdm(valid_loader)
                result = {'label': [], 'bag_score': []}
                for val_data in val_bar:
                    _, val_bags, val_labels = val_data

                    if val_labels == 0:
                        val_0 += 1
                    if val_labels == 1:
                        val_1 += 1
                    val_prob, predict_v, val_ins_prob,rank_ins = net(val_bags.to(device))

                    # weighted loss
                    weight = torch.zeros_like(val_labels).float().to(device)
                    weight = torch.fill_(weight, weight_0)
                    weight[val_labels == 1] = weight_1
                    loss_v = nn.BCELoss(weight=weight)(val_prob[0], val_labels.to(device).to(torch.float32))

                    loss_v = loss_v + reg_loss(net)

                    running_valid_loss += loss_v.item()
                    acc_val += torch.eq(predict_v, val_labels.to(device)).sum().item()

                    if predict_v == val_labels.to(device):
                        if val_labels == 1:
                            val_rt_1 += 1
                        if val_labels == 0:
                            val_rt_0 += 1
                    result['label'].append(int(val_labels))
                    result['bag_score'].append(val_prob[0][0].cpu().numpy().tolist())  # tensor to numpy to float

            fpr, tpr, thresholds = roc_curve(result['label'], result['bag_score'], pos_label=1)  # pos_label：1
            auc_val = auc(fpr, tpr)

            print('val auc:',auc_val)
            print('val right prediction 0:', val_rt_0, '/', val_0)
            print('val right prediction 1:', val_rt_1, '/', val_1)
            print('val right prediction 0:', val_rt_0, '/', val_0, file=train_text)
            print('val right prediction 1:', val_rt_1, '/', val_1, file=train_text)
            val_loss = running_valid_loss / len(valid_loader)
            val_accurate = acc_val / len(valid_loader.dataset)
            lr_scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_accurate)
            history['val_acc'].append(val_accurate)

            print('[epoch %d] train_loss: %.3f  val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_loss, train_accurate, val_accurate))
            print('[epoch %d] train_loss: %.3f  val_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_loss, train_accurate, val_accurate), file=train_text)

            # early stop
            if auc_val > best_auc:
                best_auc = auc_val
                es = 0
                torch.save(net, './evaluation/weight_auc_' + str(set_list.index(s)) + '.pth')
            else:
                es += 1
                print("Counter {} of {}".format(es, patience))
                if es > int(patience):
                    print("Early stopping with best_acc: {}, and val_acc for this epoch:{}".format(best_acc, val_accurate))
                    break

        plt.figure(figsize=(7, 7))
        plt.plot(history['train_loss'], label='Training loss')
        plt.plot(history['val_loss'], label='Validation loss')
        plt.legend()
        plt.savefig('./evaluation/loss_' + str(set_list.index(s)) + '.png')
        # plt.show()

        plt.figure(figsize=(7, 7))
        plt.plot(history['train_acc'], label='train_acc')
        plt.plot(history['val_acc'], label='valid_acc')
        plt.legend()
        plt.savefig('./evaluation/val_' + str(set_list.index(s)) + '.png')
        # plt.show()

    print('Finished Training')
