import os
import time
import torch
import argparse
import random
import warnings
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from iemocap import IEMOCAP, Transform
from models.cnn_rnn import CNN_RNN
from models.dann import DANN
#from models.center_loss import CenterLoss
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
from itertools import zip_longest

warnings.filterwarnings("ignore")
device = torch.device('cuda')


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-root_dir', type=str,
                        default='F:\singlecorpus\data\maked/')

    parser.add_argument('-exp_train', type=str, default='iemocap')
    parser.add_argument('-exp_test', type=str, default='iemocap')
    parser.add_argument('-num_classes', action="store_true", default=2)
    parser.add_argument('-train_batch_size', action="store_true", default=32)
    parser.add_argument('-num_epochs', action="store_true", default=30)
    parser.add_argument('-lr', action="store_true", default=0.001)
    args = parser.parse_args()
    return args


def train_model(model, train_loader, test_loader, epoch, optimizer):
    loss_function = nn.CrossEntropyLoss()
    model.train()
    gt_labels = []
    pred_labels = []
    total_loss = []
    center_loss = []
    for i, (train, test) in enumerate(zip_longest(train_loader, test_loader), 1):
        # print('i{}'.format(i))
        if train == None:
            break
        features, labels, gender, lengths = train

        if test == None:
            t_features, _, t_gender, t_lengths = train
        else:
            t_features, _, t_gender, t_lengths = test
        features, labels, gender, t_features, t_gender = features.to(device), labels.to(device), gender.to(device), t_features.to(device), t_gender.to(device)
        logits, logits_lantrain, logits_lantest = model(features, lengths.cpu(), t_features, t_lengths.cpu())
        ####################################
        gender_train = loss_function(logits_lantrain, gender)
        gender_test = loss_function(logits_lantest, t_gender)
        gen = gender_train + gender_test
        loss = loss_function(logits, labels)
        loss = loss + 0.2 * gen
        ##########################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            pred_labels.append(pred)
        for lab in labels.detach().cpu().numpy():
            gt_labels.append(lab)
        total_loss.append(loss.item())

    train_acc = accuracy_score(gt_labels, pred_labels)
    train_loss = np.mean(np.asarray(total_loss))
    return train_acc, train_loss


def test_model(model, loader, epoch, optimizer):
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    gt_labels = []
    pred_labels = []
    total_loss = []
    ####calculate unweighted acc####
    happy_pred = []
    happy_label = []
    angry_pred = []
    angry_label = []
    sad_pred = []
    sad_label = []
    neu_pred = []
    neu_label = []
    ####calculate unweighted acc####

    with torch.no_grad():
        for i, (features, labels, gender, lengths) in enumerate(loader, 1):
            features, labels = features.to(device), labels.to(device)
            logits, _, _ = model(features, lengths.cpu(), features, lengths.cpu())
            loss = loss_function(logits, labels.squeeze())
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                pred_labels.append(pred)
            for lab in labels.detach().cpu().numpy():
                gt_labels.append(lab)
            total_loss.append(loss.item())

            ####calculate unweighted acc####
            for k in range(len((labels))):
                lab_emo = labels[k]
                pred_emo = predictions[k]
                if lab_emo == 0:
                    neu_label.append(lab_emo.detach().cpu().numpy().item())
                    neu_pred.append(pred_emo)
                elif lab_emo == 1:
                    angry_label.append(lab_emo.detach().cpu().numpy().item())
                    angry_pred.append(pred_emo)
                elif lab_emo == 2:
                    sad_label.append(lab_emo.detach().cpu().numpy().item())
                    sad_pred.append(pred_emo)
                else:
                    happy_label.append(lab_emo.detach().cpu().numpy().item())
                    happy_pred.append(pred_emo)

    accuracy_happy = accuracy_score(happy_label, happy_pred)
    accuracy_angry = accuracy_score(angry_label, angry_pred)
    accuracy_sad = accuracy_score(sad_label, sad_pred)
    accuracy_neu = accuracy_score(neu_label, neu_pred)
    test_uw_acc = np.mean([accuracy_happy, accuracy_angry, accuracy_sad, accuracy_neu])
    ####calculate unweighted acc####

    test_w_acc = accuracy_score(gt_labels, pred_labels)
    test_loss = np.mean(np.asarray(total_loss))

    return test_loss, test_w_acc, test_uw_acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(exp):
    args = parse_args()
    setup_seed(199633)

    best_wa, best_epoch_wa = 0, 0
    best_ua, best_epoch_ua = 0, 0

    train_db = IEMOCAP(args.root_dir, experiments= exp, is_all_sample=True, train_or_test="train",
                       transform=Transform())
    test_db = IEMOCAP(args.root_dir, experiments= exp, is_all_sample=True, train_or_test="test",
                      transform=Transform())
    train_db.next()
    test_db.next()
    train_loader = DataLoader(train_db, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=args.train_batch_size, shuffle=False)

    model = DANN().to(device)
    # summary(model, (1, 32, 128))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_acc, train_loss = train_model(model, train_loader, test_loader, epoch, optimizer)
        test_loss, test_wa, test_ua = test_model(model, test_loader, epoch, optimizer)
        if test_ua > best_ua:
            best_ua = test_ua
        if test_wa > best_wa:
            #best_epoch_ua = epoch
            best_wa = test_wa
            #torch.save(model.state_dict(), './save_models/bestmodel_wa.hdf5')

        end_time = time.time()
        print("epoch:{} time:{:.2f}min train_loss:{:.2f} train_wa:{:.2%} "
              "test_loss:{:.2f} test_wa:{:.2%} test_ua:{:.2%}".format(epoch, (end_time - start_time) / 60,
                                                                      train_loss, train_acc,
                                                                      test_loss, test_wa, test_ua))

    print('trainon{} best wa_acc{:.2%}: best ua_acc:{:.2%}'.format(exp, best_wa, best_ua))
    print('\n')



if __name__ == '__main__':
    for exp in range(1, 6):
        run(str(exp))
