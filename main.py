
import os
import copy
import cv2
import pandas as pd
import numpy as np
import csv
import logging
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import performances_compute
from backbones import mixnet_s

device = torch.device('cuda:0')

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]
INPUT_SIZE = 224
EarlyStopPatience = 20

class FaceDataset(Dataset):
    def __init__(self, file_name, is_train):
        self.data = pd.read_csv(file_name)
        self.is_train = is_train
        self.train_transform = transforms.Compose(
            [
             transforms.ToPILImage(),
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=PRE__MEAN,
                                 std=PRE__STD),
             ])

        self.test_transform = transforms.Compose(
            [           transforms.ToPILImage(),
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
             transforms.ToTensor(),
             transforms.Normalize(mean=PRE__MEAN,
                                 std=PRE__STD),
             ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label_str = self.data.iloc[index, 1]
        label = 1 if label_str == 'bonafide' else 0

        image=cv2.imread(image_path)
        try:
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.test_transform(image)
        except ValueError:
            print(image_path)

        return image, label

def train_fn(model, data_loader, data_size, optimizer, criterion):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects.double() / data_size

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

def eval_fn(model, data_loader, data_size, criterion):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        prediction_scores, gt_labels = [], []
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss= criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            probs = F.softmax(outputs, dim=1)
            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i][1].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        _, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)

        print('{} Loss: {:.4f} Acc: {:.4f} EER: {:.4f}'.format('Val', epoch_loss, epoch_acc, eer_value))

    return epoch_loss, epoch_acc, eer_value

def run_training(model, model_path, logging_path, normedWeights, num_epochs, dataloaders, dataset_sizes):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=normedWeights).to(device)
    optimizer=optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

    logging.basicConfig(filename=logging_path, level=logging.INFO)

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_eer = 100
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        # Each epoch has a training and validation phase
        train_loss, train_acc = train_fn(model, dataloaders['train'], dataset_sizes['train'], optimizer, criterion)
        val_loss, val_acc, val_eer_values = eval_fn(model, dataloaders['val'], dataset_sizes['val'], criterion)
        logging.info('train loss: {}, train acc: {}, val loss: {}, val acc: {}, val eer: {}'.format(train_loss, train_acc, val_loss, val_acc, val_eer_values))

        # deep copy the model
        if val_eer_values <= lowest_eer:
            lowest_eer = val_eer_values
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == EarlyStopPatience or epoch >= num_epochs:
            early_stop = True
        else:
            continue

        if early_stop:
            print('Trian process Stopped')
            print('epoch: {}'.format(epoch))
            break

    print('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info('Lowest EER: {:4f}'.format(lowest_eer))
    logging.info(f'saved model path: {model_path}')

    # save best model weights
    torch.save(best_model_wts, model_path)

def run_test(test_loader, model, model_path, batch_size=64):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction_scores, gt_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i][1].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

        std_value = np.std(prediction_scores)
        mean_value = np.mean(prediction_scores)
        prediction_scores = [ (float(i) - mean_value) /(std_value) for i in prediction_scores]
        _, eer_value, _ = performances_compute(prediction_scores, gt_labels, verbose=False)
        print(f'Test EER value: {eer_value*100}')

    return prediction_scores

def write_scores(test_csv, prediction_scores, output_path):
    save_data = []
    dataframe = pd.read_csv(test_csv)
    for idx in range(len(dataframe)):
        image_path = dataframe.iloc[idx, 0]
        label = dataframe.iloc[idx, 1]
        label = label.replace(' ', '')
        save_data.append({'image_path': image_path, 'label':label, 'prediction_score': prediction_scores[idx]})

    with open(output_path, mode='w') as csv_file:
        fieldnames = ['image_path', 'label', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in save_data:
            writer.writerow(data)

    print(f'Saving prediction scores in {output_path}.')

def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # model path that will used to save the trained model or where is the pre-trained weights
    #model_path = os.path.join(args.output_dir, 'mixfacenet_SMDD')
    model = mixnet_s(embedding_size=128, width_scale=1.0, gdw_size=1024, shuffle=False)

    if args.is_train:
        train_dataset = FaceDataset(args.train_csv_path, is_train=True)
        test_dataset = FaceDataset(args.test_csv_path, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        dataloaders = {'train': train_loader, 'val': test_loader}
        dataset_sizes = {'train': len(train_dataset), 'val': len(test_loader)}
        print('train and test length:', len(train_dataset), len(test_loader))

        # compute loss weights to improve the unbalance between data
        attack_num, bonafide_num = 0, 0
        with open(args.train_csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['label'] == 'attack':
                    attack_num += 1
                else:
                    bonafide_num += 1
        print('attack and bonafide num:', attack_num, bonafide_num)

        nSamples  = [attack_num, bonafide_num]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(device)

        #create log file and train model
        logging_path = os.path.join(args.output_dir, 'train_info.log')
        run_training(model, args.model_path, logging_path, normedWeights, args.max_epoch, dataloaders, dataset_sizes)

    if args.is_test:
        # create save directory and path
        test_output_folder = os.path.join(args.output_dir, 'test_results')
        Path(test_output_folder).mkdir(parents=True, exist_ok=True)
        test_output_path = os.path.join(test_output_folder, 'test_results.csv')
        # test
        test_dataset = FaceDataset(file_name=args.test_csv_path, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_prediction_scores = run_test(test_loader=test_loader, model=model, model_path=args.model_path)
        write_scores(args.test_csv_path, test_prediction_scores, test_output_path)

if __name__ == '__main__':

    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='MixFaceNet model')
    parser.add_argument("--train_csv_path", default="dataset_info/train.csv", type=str, help="input path of train csv")
    parser.add_argument("--test_csv_path", default="dataset_info/test.csv", type=str, help="input path of test csv")

    parser.add_argument("--output_dir", default="output", type=str, help="path where trained model and test results will be saved")
    parser.add_argument("--model_path", default="mixfacenet_SMDD", type=str, help="path where trained model will be saved or location of pretrained weight")

    parser.add_argument("--is_train", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="train database or not")
    parser.add_argument("--is_test", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="test database or not")

    parser.add_argument("--max_epoch", default=100, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")

    args = parser.parse_args()

    main(args)
