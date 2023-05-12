import csv
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# from my bag
from dataloader import read_bag, aaMILDataset

if __name__ == '__main__':

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    csv_path = 'label/fold_cesc_t2.csv'
    data_path = 'dataset/cesc_t2_png_224'

    # means, stds = nom_para(data_path)
    means = [0.15708263, 0.15708263, 0.15708263] # t2
    stds = [0.13905464, 0.13905464, 0.13905464]

    x = 1000 # modality num, T2:1000
    input_size = 224
    result_text = open("./evaluation/result_slice.txt", "a")
    test_text = open("./evaluation/test_slice.txt", "a")
    nw = 16

    data_transform = {
        "valid": transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((means[0], means[1], means[2]), (stds[0], stds[1], stds[2]))
                                     ])}

    csvfile = open(csv_path,encoding='UTF-8-sig')
    set_csv = csv.reader(csvfile)
    set_list = []

    acc_l = []
    auc_l = []
    precision_l = []
    recall_l = []
    specificity_l = []
    f1_l = []
    tpr_l = []
    fpr_m = np.linspace(0, 1, 100)

    for i, rows in enumerate(set_csv): # i: set number, rows: ["['3', 0]", "['30', 1]"]
        set_list.append(rows)

    for s in set_list:
        test_list = s # ["['9', 0]", "['28', 1]"]
        weights_path = './evaluation/weight_auc_0.pth'
        model = torch.load(weights_path)

        test_patient_list, test_label_list = read_bag(test_list, x, data_path)
        test_df = pd.DataFrame({'bag': test_patient_list, 'label': test_label_list})
        test_dataset = aaMILDataset(test_df, transform=data_transform['valid'])
        test_num = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle = True, num_workers=nw)
        print("Fold:{}, test_patient:{}".format(str(set_list.index(s)), test_num))
        print("Fold:{}, test_patient:{}".format(str(set_list.index(s)), test_num), file=result_text)

        result = {'patient': [], 'predict': [], 'label': [], 'bag_score': [], 'ins_score': [],'rank_prob':[]}

        # validate
        model.eval()
        with torch.no_grad():
            for test_data in test_loader:
                bag_p, test_bags, test_labels = test_data
                bag_prob,predict_t,ins_prob,rank_ins_prob = model(test_bags.to(device))

                result['patient'].append(bag_p[0].replace(data_path,''))
                result['label'].append(int(test_labels))
                result['bag_score'].append(bag_prob[0][0].cpu().numpy().tolist()) # tensor to numpy to float
                result['ins_score'].append(ins_prob)
                result['rank_prob'].append(rank_ins_prob)

        # best threshold
        fpr, tpr, thresholds = roc_curve(result['label'], result['bag_score'], pos_label=1)  # pos_label：1
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        best_threshold = thresholds[maxindex]
        print(best_threshold)

        for i in result['bag_score']:
            if i >= best_threshold:
                result['predict'].append(1)
            else:
                result['predict'].append(0)

        for i in range(len(result['patient'])):
            print("patient:{}, label:{}, predict:{}, bag_prob:{}, ins_prob:{},rank_prob:{}".
                  format(result['patient'][i], result['label'][i], result['predict'][i],result['bag_score'][i], result['ins_score'][i],
                         result['rank_prob'][i]),
                  file=result_text
                  )

        cm = confusion_matrix(result['label'], result['predict']) # cm.ravel(): tn,fp,fn,tp
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(result['label'], result['predict'])
        acc_l.append(acc)
        precision = precision_score(result['label'], result['predict'])
        precision_l.append(precision)
        recall = recall_score(result['label'], result['predict'])
        recall_l.append(recall)
        f1 = f1_score(result['label'], result['predict']) # f1
        f1_l.append(f1)
        specificity = tn /(fp + tn)
        specificity_l.append(specificity)

        tpr_l.append(interp(fpr_m, fpr, tpr))
        tpr_l[-1][0] = 0.0
        rauc = auc(fpr, tpr) # auc
        auc_l.append(rauc)

        print("Fold:{}, acc:{}, auc:{}, pre:{}, recall:{}, spec:{}, f1:{}".format(str(set_list.index(s)), acc, rauc, precision, recall, specificity, f1))
        print("Fold:{}, acc:{:.3f}, auc:{:.3f}, pre:{:.3f}, recall:{:.3f}, spec:{:.3f}, f1:{:.3f}".format(str(set_list.index(s)), acc, rauc, precision, recall, specificity, f1), file=test_text)

        plt.plot(fpr, tpr, label='AUC {}-fold = {:.3f}'.format(set_list.index(s),rauc))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.title('ROC curve_fold {:.3f}'.format(set_list.index(s)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
