import time
import os
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from BCN_pred import BCNN
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm, trange
from Early_Stopping import EarlyStopping
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

parser = argparse.ArgumentParser(description='Get the input arguments')
parser.add_argument('--gpuid', default=0, type=int,
                    help='choose the gpu u are going to use, otherwise cpu')

def Preprocess(label) :
    dataLabelsFile = r'./data/ADNI_GO_2.csv'
    dataLabels = np.loadtxt(dataLabelsFile, str, delimiter=",", encoding='utf-8-sig', skiprows=1)
    labelDict = {
        'Image Data ID' : 0,
        'Subject' : 1,
        'Group' : 2,
        'Sex' : 3,
        'Age' : 4,
        'Visit' : 5,
        'Modality' : 6,
        'Description' : 7,
        'Type' : 8,
        'Acq Date' : 9,
        'Format' : 10,
        'Downloaded' : 11
    }
    dataFiles = []
    dataLabelFiles = []
    for dataLabel in dataLabels :
        subjectOldTime = time.strptime(dataLabel[labelDict['Acq Date']], "%m/%d/%Y")
        subjectNewTime = time.strftime("%Y%m%d", subjectOldTime)
        dataFile = "./data/FC_264/" + dataLabel[labelDict['Subject']] + "_" + subjectNewTime + ".npy"
        if os.path.isfile(dataFile) and dataFile not in dataFiles :
            if dataLabel[labelDict[label]] == "CN" or dataLabel[labelDict[label]] == "AD" :
                dataFiles.append(dataFile) 
                dataLabelFiles.append([dataFile, dataLabel[labelDict[label]]])
            # print(dataFile, dataLabel[labelDict[label]])
    return dataLabelFiles

class Dataset_Sia_(Dataset):
    """
    load nifiti image as dataset {image: value, label: label}
    """
    def __init__(self, root_dir, img_name, img_label, transformer):
        self.root_dir = root_dir
        self.img_name = img_name
        self.img_label = img_label
        self.transformer = transformer

    def __len__(self):
        return sum(1 for line in self.img_name)

    def __getitem__(self, idx):

        # filename = 'sub-' + self.img_name[idx] + '.mat'
        filename = self.img_name[idx]
        # filepath = os.path.join(self.root_dir, filename)

        img = np.load(filename)
        # img = np.load(filepath)
        # img = loadmat(filename)['fc']
        # img = np.expand_dims(img, axis=2)
        # img = np.concatenate((img, img, img), axis=-1)
        # img = img.transpose(2, 0, 1)
        # if int(self.img_label[idx]) == 1:
        #     label = 1
        # elif int(self.img_label[idx]) == 2:
        #     label = 0
        label = round(float(self.img_label[idx]))
        img = self.transformer(img)

        return img, label

def train_loop(i_fold, model, loss_fc, optimizer, train_loader, val_loader, device):
    best_acc = None
    train_loss_li, val_acc_li, val_loss_li = [], [], []
    r2_li = []
    patience = 100
    es = EarlyStopping(patience, verbose=False)
    # modelAE.load_state_dict(torch.load(args.saveAE + 'epoch_{}_model.pth'.format(args.aepoch)))
    # modelAE.eval()
    model.train()
    with trange(epoch) as t:
        for i in t:
            t.set_description("Epoch %i" % i)
            train_out, val_out = [], []
            train_label, val_label = [], []
            train_loss, val_loss = 0.0, 0.0

            for img, label in train_loader:
                # fc = []
                # print(img.shape)
                img = img.to(device).float()
                label = label.to(device).float()
                # label = label.squeeze(1)
                # for j in img:
                #     fc.append(j.to(device, dtype=torch.float))
                # input = modelAE(img)

                output = model(img)
                # print(output.size(), label.size())
                loss = loss_fc(output.squeeze(), label)
                train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            for img, label in val_loader:
                with torch.no_grad():
                    img = img.to(device).float()
                    label = label.to(device).float()
                    val_label.extend(label.cpu())
                    # input = modelAE(img)
                    output = model(img)
                    # print(predict.squeeze(1), label)

                    loss_val = loss_fc(output.squeeze(), label.float())
                    val_loss += loss_val
                    # print(output)
                    val_out.extend(output.squeeze().cpu())
            # val_out = torch.stack(val_out, 0)
            # print(val_out)
            # val_label = torch.cat(val_label).reshape(-1)
            avg_loss_val = val_loss / len(val_loader) / batch_size
            # feat_list = np.array(val_out)
            # label_list = np.array(val_label)

            # r2 = r2_score(feat_list, label_list)
            # r = pearsonr(feat_list, label_list)
            
            # mae = mean_absolute_error(label_list, feat_list)
            # if (i+1) % 50 == 0 or i == 0:
            #     r2_li.append(r[0])
            # t.set_postfix(loss_reg='{:.6f}'.format(avg_loss_val.item()),
            #               R='{:.4f}'.format(r[0]), MAE='{:.4f}'.format(mae), lst=["BCNN dCor age NKI pred"])
            models = [model]
            es(avg_loss_val, models, save)
            if es.early_stop and avg_loss_val < 0.1:
                print("STOP!")
                break
    return r2_li

def r2_s(y_test, y_pred):
    mse_test = np.sum((y_test - y_pred) ** 2) / len(y_test)
    var = np.var(y_test)
    r2 = 1 - mse_test / var
    return r2

def plot_regression(y_, result, avg_r, avg_mae):
    axismin, axismax = 60, 90
    y_ = np.array(y_)
    result = np.array(result)
    [k, b] = np.polyfit(y_, result.reshape(-1, 1), 1)               #计算斜率和截距
    # print(k, b)
    x_axis = np.linspace(axismin + 0.01 * axismax, axismax * 0.99, 69696)
    # y_pred = model_regress.predict(x_axis)
    y_axis = k * x_axis + b
    y_regress = np.polyval([k, b], x_axis)

    plt.figure()
    plt.scatter(y_, result, c='none', marker='o',edgecolors='k', s=10)


    plt.plot(x_axis, y_regress, 'r-', label='output fit line', lw=3.0)
    # # ax = seaborn.regplot(x=result, y=y_, ci=90)
    plt.xlim(axismin, axismax)
    plt.ylim(axismin, axismax)
    plt.text(84, 89, 'R={:.3f}'.format(avg_r), fontweight='bold')
    plt.text(84, 88, 'MAE={:.3f}'.format(avg_mae), fontweight='bold')
    plt.xlabel('Actual age')
    plt.ylabel('Predicted age')
    plt.title('Age Regression')
    plt.legend(loc='upper left')
    savename = 'BCNN_age_dCor_NKI.jpg'
    plt.savefig(savename)





if __name__ ==  '__main__':
    args = parser.parse_args()
    KFolds = 10
    batch_size = 1024
    gpuid = args.gpuid
    lr = 1e-3
    wd = 1e-5
    epoch = 1000
    save = './best_model_pred/'

    device = torch.device('cuda:{}'.format(gpuid))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    datasets = Preprocess("Group")
    Input_path = './data/FC_264/'
    all_df, all_labels = [], []

    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])


    for dataset in datasets :
        # data = np.load(dataset[0])
        # label = dataset[1]
        # print(dataset)
        all_df.append(dataset[0])
        # all_labels.append(dataset[1])
        if dataset[1] == 'CN':
            all_labels.append(0)
        else :
            all_labels.append(1)

    folder = StratifiedKFold(n_splits=KFolds, shuffle=True)
    sum_r, sum_r2, sum_mae = 0.0, 0.0, 0.0
    res_arr, label_arr, score_arr = [], [], []
    str1 = '=' * 20
    r_sum = []
    count = 0
    print('{0}Start Cross Validation{1}'.format(str1, str1))
    accuracy, precision, recall, f1 = [], [], [] ,[]


    for num, (train_idx, test_idx) in enumerate(folder.split(all_df, all_labels)):
        print('Folds:{0}/{1}'.format(num + 1, KFolds))
        test_set, test_label = [], []
        train_set, train_label = [], []
        valid_set, valid_label = [], []
        test_out, test_l = [], []
        train_val_n = len(train_idx)
        temp_idx = train_idx
        random.shuffle(temp_idx)
        train_idx = temp_idx[:int(train_val_n * 0.9)]
        valid_idx = temp_idx[int(train_val_n * 0.9):]

        """GAN"""
        for generate_data_count in range(0, 1000):
            if (os.path.isfile("./generate_data/data/subject_"  + str(generate_data_count).zfill(3) + ".npy")) :
                train_set.append("./generate_data/data/subject_"  + str(generate_data_count).zfill(3) + ".npy")
                tmp_label = np.load("./generate_data/label/label_"  + str(generate_data_count).zfill(3) + ".npy")
                train_label.append(tmp_label[0])
        # for generate_data_count in range(901, 1000):
        #     if (os.path.isfile("./generate_data/data/subject_"  + str(generate_data_count).zfill(3) + ".npy")) :
        #         valid_set.append("./generate_data/data/subject_"  + str(generate_data_count).zfill(3) + ".npy")
        #         tmp_label = np.load("./generate_data/label/label_"  + str(generate_data_count).zfill(3) + ".npy")
        #         valid_label.append(tmp_label[0])
        """GAN"""

        test_set.append([all_df[idx] for idx in test_idx])
        test_label.append([all_labels[idx] for idx in test_idx])
        test_set = np.array(test_set).squeeze()
        test_label = np.array(test_label).squeeze()

        # train_set.append([all_df[idx] for idx in train_idx])
        # train_label.append([all_labels[idx] for idx in train_idx])
        train_set += [all_df[idx] for idx in train_idx]
        train_label += [all_labels[idx] for idx in train_idx]
        train_set = np.array(train_set).squeeze()
        train_label = np.array(train_label).squeeze()

        # valid_set += [all_df[idx] for idx in valid_idx]
        # valid_label += [all_labels[idx] for idx in valid_idx]
        valid_set.append([all_df[idx] for idx in valid_idx])
        valid_label.append([all_labels[idx] for idx in valid_idx])
        val_set = np.array(valid_set).squeeze()
        val_label = np.array(valid_label).squeeze()

        # train_dataset = Dataset_Sia_(Input_path, train_set, train_label, transformer)
        # val_dataset = Dataset_Sia_(Input_path, val_set, val_label, transformer)
        # test_dataset = Dataset_Sia_(Input_path, test_set, test_label, transformer)
        train_datas, val_datas, test_datas = [], [], []
        for train_df in train_set:
            train_datas.append([np.load(train_df)])
        for val_df in val_set:
            val_datas.append([np.load(val_df)])  
        for test_df in test_set :
            test_datas.append([np.load(test_df)])

        train_datas = np.array(train_datas, dtype=float)
        val_datas = np.array(val_datas, dtype=float)
        test_datas = np.array(test_datas, dtype=float)
        train_label = np.array(train_label, dtype=float)
        val_label = np.array(val_label, dtype=float)
        test_label = np.array(test_label, dtype=float)

        train_datas = torch.from_numpy(train_datas).type(torch.FloatTensor)
        val_datas = torch.from_numpy(val_datas).type(torch.FloatTensor)
        test_datas = torch.from_numpy(test_datas).type(torch.FloatTensor)
        train_label = torch.from_numpy(train_label).type(torch.FloatTensor)
        val_label = torch.from_numpy(val_label).type(torch.FloatTensor)
        test_label = torch.from_numpy(test_label).type(torch.FloatTensor)

        train_dataset = TensorDataset(train_datas, train_label)
        val_dataset = TensorDataset(val_datas, val_label)
        test_dataset = TensorDataset(test_datas, test_label)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=0,
                                drop_last=False,
                                pin_memory = True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=0,
                                drop_last=False,
                                pin_memory = True)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                num_workers=0,
                                drop_last=False,
                                pin_memory = True)
        # print(len(train_loader.indices))
        model = BCNN(e2e=16, e2n=512, n2g=128, f_size=264, dropout=0.5)
        model.to(device)
        # loss_fc = nn.MSELoss().to(device)
        loss_fc = nn.BCEWithLogitsLoss().to(device)
        lr = lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        r_li = train_loop(num, model, loss_fc, optimizer, train_loader, val_loader, device)

        model.eval()
        # model.load_state_dict(torch.load(save + 'model_BCNN_dCor_age_NKI.pt'))
        for fc, label in test_loader:
            with torch.no_grad():
                fc = fc.to(device).float()
                # label = torch.from_numpy(np.array(label)).to(device)
                label = label.to(device).float()
                test_l.extend(label.cpu())
                # input = modelAE(fc)
                output = model(fc)
                test_out.extend(output.squeeze().cpu())

        feat_list = np.array(test_out)
        label_list = np.array(test_l)
        feat_score = torch.sigmoid(torch.from_numpy(feat_list)).numpy()
        feat_list = torch.round(torch.sigmoid(torch.from_numpy(feat_list))).numpy()
        
        print(feat_list)

        accuracy.append(accuracy_score(feat_list, label_list))
        precision.append(precision_score(feat_list, label_list, zero_division = 1))
        recall.append(recall_score(feat_list, label_list, zero_division = 1))
        f1.append(f1_score(feat_list, label_list, zero_division = 1))
        
        print("accuracy : %.6f" % accuracy_score(feat_list, label_list))
        print("precision : %.6f" % precision_score(feat_list, label_list, zero_division = 1))
        print("recall : %.6f" % recall_score(feat_list, label_list, zero_division = 1))
        print("f1 : %.6f" % f1_score(feat_list, label_list, zero_division = 1))

        score_arr.extend(feat_score)
        res_arr.extend(feat_list)
        label_arr.extend(label_list)
    print("accuracy : %.6f" % np.mean(accuracy))
    print("precision : %.6f" % np.mean(precision))
    print("recall : %.6f" % np.mean(recall))
    print("f1 : %.6f" % np.mean(f1))

    np.save('bcnn_classification_gan_feat_array.npy', np.hstack(res_arr))
    np.save('bcnn_classification_gan_real_array.npy', np.hstack(label_arr))
    np.save('bcnn_classification_gan_score_array.npy', np.hstack(score_arr))
