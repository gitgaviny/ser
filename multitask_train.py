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
from models.multitask import Multi_task
from models.cnn_rnn import CNN_RNN
from models.cnn_rnn_att import CNN_RNN_ATT
from sklearn.metrics import classification_report, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
warnings.filterwarnings("ignore")
device = torch.device('cuda')


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-root_dir', type=str,
                        default='/Work20/2019/gaoyuan/baseline/iemocap_data/')
    #parser.add_argument('-experiments', type=str, default='5')
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-save_dir', type=str, default='save_models')

    parser.add_argument('-num_classes', action="store_true", default=4)
    parser.add_argument('-train_batch_size', action="store_true", default=16)
    parser.add_argument('-num_epochs', action="store_true", default=30)
    parser.add_argument('-lr', action="store_true", default=0.001)
    args = parser.parse_args()
    return args


def train_model(model, loader, epoch, optimizer):
    loss_function = nn.CrossEntropyLoss()
    model.train()
    gt_labels = []
    pred_labels = []
    total_loss = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, min_lr=0.0001,
                                                           patience=3)
    for i, (features, labels, gender, lengths) in enumerate(loader, 1):
        features, labels, gender = features.to(device), labels.to(device), gender.to(device)
        emo_logits, gen_logits = model(features, lengths.cpu())
        emo_loss = loss_function(emo_logits, labels.squeeze())
        gen_loss = loss_function(gen_logits, gender.squeeze())
        loss = emo_loss + 0.5 * gen_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            pred_labels.append(pred)
        for lab in labels.detach().cpu().numpy():
            gt_labels.append(lab)
        total_loss.append(loss.item())

    train_acc = accuracy_score(gt_labels, pred_labels)
    train_loss = np.mean(np.asarray(total_loss))
    scheduler.step(train_acc)
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
        for i, (features, labels, gender, lengths) in enumerate(loader, 0):
            features, labels, gender = features.to(device), labels.to(device), gender.to(device)
            emo_logits, gen_logits = model(features, lengths.cpu())
            emo_loss = loss_function(emo_logits, labels.squeeze())
            gen_loss = loss_function(gen_logits, gender.squeeze())
            loss = emo_loss + 0.5 * gen_loss
            predictions = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
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


def run(exp=1):
    args = parse_args()
    setup_seed(1234)
    best_wa, best_epoch_wa = 0, 0
    best_ua, best_epoch_ua = 0, 0

    train_db = IEMOCAP(args.root_dir, experiments= exp, is_all_sample=True, train_or_test="train",
                       transform=Transform())
    test_db = IEMOCAP(args.root_dir, experiments= exp, is_all_sample=True, train_or_test="test",
                      transform=Transform())
    train_db.next()
    test_db.next()
    train_loader = DataLoader(train_db, batch_size=args.train_batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=args.train_batch_size, num_workers=8, shuffle=False)

    model = Multi_task().to(device)
    # summary(model, (1, 32, 128))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_acc, train_loss = train_model(model, train_loader, epoch, optimizer)
        test_loss, test_wa, test_ua = test_model(model, test_loader, epoch, optimizer)
        if test_ua > best_ua:
            #best_epoch_ua = epoch
            best_ua = test_ua
            #torch.save(model.state_dict(), './save_models/bestmodel_ua.hdf5')
        if test_wa > best_wa:
            #best_epoch_ua = epoch
            best_wa = test_wa
            #torch.save(model.state_dict(), './save_models/bestmodel_wa.hdf5')

        end_time = time.time()
        print("epoch:{} time:{:.2f}min train_loss:{:.2f} train_wa:{:.2%} "
              "test_loss:{:.2f} test_wa:{:.2%} test_ua:{:.2%}".format(epoch, (end_time - start_time) / 60,
                                                          train_loss, train_acc,
                                                          test_loss, test_wa, test_ua))

    print('testonsession{} best wa_acc{:.2%}: best ua_acc:{:.2%}'.format(exp, best_wa, best_ua))
    print('\n\n')


if __name__ == '__main__':
    for exp in range(1,6):
        run(str(exp))
