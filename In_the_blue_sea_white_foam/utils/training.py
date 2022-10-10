import numpy as np
import pandas as pd
import torch
from torch import nn


def train_one_epoch(model, train_dataloader, device, optimizer, loss, metric, iepoch, nepochs, nextend, log_each=20):
    # До этого определили размер батча для train_dataloader
    # Теперь надо пройти по всем батчам картинок для обучения одной эпохи
    nbatches = len(train_dataloader)
    total_loss_value = 0.0
    if metric:
        total_metric_value = 0.0
    count = 0

    # training step
    model.train()

    for iextend in range(nextend):
        for ibatch, sample_batch in enumerate(train_dataloader):
            images, labels_true = sample_batch['image'], sample_batch['label']
            images = images.to(device)
            labels_true = labels_true.to(device)
            n = labels_true.shape[0]
            count += n
            optimizer.zero_grad()
            enc_labels_pred = model(images)
            # enc_labels_pred = model.forward(images)
            loss_value = loss(enc_labels_pred, labels_true)
            total_loss_value += loss_value.item() * n
            loss_value.backward()
            optimizer.step()

            if metric:
                # switch to evaluation mode
                model.eval()
                with torch.no_grad():
                    if type(metric) is not nn.CrossEntropyLoss:
                        # Метрики принимают на вход вероятностные вектора
                        enc_labels_pred = nn.Softmax()(enc_labels_pred)
                    metric_value = metric(enc_labels_pred, labels_true).item()
                    total_metric_value += metric_value * n
                model.train()

            if (iextend * nbatches + ibatch + 1) % log_each == 0:
                print(
                    f"(Epoch {iepoch + 1}/{nepochs}-{iextend * nbatches + ibatch + 1}/{nbatches * nextend})\t",
                    f"Loss: {loss_value.item():.6f}\t", sep='', end=''
                )
                if metric:
                    print(f"Metric: {metric_value:.6f}", end='')
                print()
    total_loss_value /= count
    print(
        f"Resulrts model training for epoch {iepoch + 1}/{nepochs}:\n\t",
        f"Loss\t: {total_loss_value:.6f}\t", sep='', end=''
    )

    if metric:
        total_metric_value /= count
        print(f"Metric\t: {total_metric_value:.6f}")
        return total_loss_value, total_metric_value
    else:
        print()

    return total_loss_value


def validate(model, valid_dataloader, device, loss, metric, nextend):
    nbatches = len(valid_dataloader)
    total_val_loss_value = 0.0
    if metric:
        total_val_metric_value = 0.0
    count = 0

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for iextend in range(nextend):
            for ibatch, sample_batch in enumerate(valid_dataloader):
                images, labels_true = sample_batch['image'], sample_batch['label']
                images = images.to(device)
                labels_true = labels_true.to(device)
                n = labels_true.shape[0]
                count += n
                enc_labels_pred = model(images)
                #  enc_labels_pred = model.forward(images)
                total_val_loss_value += loss(enc_labels_pred, labels_true).item() * n
                if metric:
                    if type(metric) is not nn.CrossEntropyLoss:
                        # Метрики принимают на вход вероятностные вектора
                        enc_labels_pred = nn.Softmax()(enc_labels_pred)
                    total_val_metric_value += metric(enc_labels_pred, labels_true).item() * n

        total_val_loss_value /= count
        print(f'\tLoss val: {total_val_loss_value:.6f}\t', end='')

        if metric:
            total_val_metric_value /= count
            print(f'Metric val: {total_val_metric_value:.6f}')
            return total_val_loss_value, total_val_metric_value
        else:
            print()
        return total_val_loss_value


def get_metric_name(metric):
    isaverage = "average" in dir(metric)
    if isaverage:
        return f"{metric._get_name()} ({metric.average})"
    else:
        return metric._get_name()


def evaluate(model, dataloader, metrics_list, device, nclasses, nextend):
    nsamples = 0
    nsamples_by_classes = np.zeros((nclasses, 1))
    metrics_class1_list = [
        metric
        for metric in metrics_list
        if ("average" not in dir(metric)) or \
           ("average" in dir(metric) and metric.average is not None)
    ]
    metrics_class2_list = [
        metric
        for metric in metrics_list
        if "average" in dir(metric) and metric.average is None
    ]

    if len(metrics_class2_list) > 0:
        accuracy_by_classes = np.zeros((len(metrics_class2_list), nclasses))

    total_metrics_value_t = torch.zeros(1, len(metrics_class1_list)).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for iextend in range(nextend):
            for ibatch, sample_batch in enumerate(dataloader):
                images, labels_true = sample_batch['image'], sample_batch['label']
                images = images.to(device)
                labels_true = labels_true.to(device)
                enc_labels_pred = model(images)
                # Метрики принимают на вход вероятностные вектора
                enc_labels_pred_softmax = nn.Softmax()(enc_labels_pred)
                n = labels_true.shape[0]
                nsamples += n

                for idx, metric in enumerate(metrics_class1_list):
                    if type(metric) is not nn.CrossEntropyLoss:
                        metric_val = metric(enc_labels_pred_softmax, labels_true).item()
                        total_metrics_value_t[0][idx] += metric_val * n
                    else:
                        total_metrics_value_t[0][idx] += metric(enc_labels_pred, labels_true).item() * n
                    # print(metric._get_name(), total_metrics_value_t[0][idx])
                if len(metrics_class2_list) > 0:
                    labels, counts = np.unique(
                        labels_true.to("cpu").numpy(),
                        return_counts=True,
                        axis=None
                    )
                    for ilabel, ncount in zip(labels, counts):
                        nsamples_by_classes[ilabel][0] += ncount

                    for jdx, metric in enumerate(metrics_class2_list):
                        metric_vals = metric(enc_labels_pred_softmax, labels_true).to("cpu").numpy()
                        # print(metric._get_name(), metric.average, metric_vals, metric_vals.shape)
                        for ilabel, ncount in zip(labels, counts):
                            accuracy_by_classes[jdx][ilabel] += metric_vals[ilabel] * ncount

                        # Бывает, что в батче нет представителей класса
                        # В таких случаях возвращается nan

        total_metrics_class1_value_s = pd.DataFrame(
            data=total_metrics_value_t.to('cpu').numpy(),
            columns=[get_metric_name(metric) for metric in metrics_class1_list]
        ).iloc[0]
        total_metrics_class1_value_s /= nsamples

        if len(metrics_class2_list) > 0:
            total_metrics_class2_value_df = pd.DataFrame(
                data=accuracy_by_classes.T,
                columns=[metric._get_name() for metric in metrics_class2_list]
            )
            total_metrics_class2_value_df /= nsamples_by_classes
            return total_metrics_class1_value_s, total_metrics_class2_value_df

        return total_metrics_class1_value_s


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(prediction, target):
    with torch.no_grad():
        prediction = torch.max(prediction, dim=1)[1]
        correct = prediction.eq(target)
        return correct.float().mean()


def avarage_evaluate(model, dataloader, device, criterion):
    # switch to evaluation mode
    model.eval()

    loss_tracker = AverageMeter()
    accuracy_tracker = AverageMeter()
    with torch.no_grad():
        for i, sample_batch in enumerate(dataloader):
            images, target = \
                sample_batch['image'], sample_batch['label']
            n_images = images.shape[0]
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            loss_value = loss.item()
            accuracy_value = accuracy(output, target).item()
            loss_tracker.update(loss_value, n_images)
            accuracy_tracker.update(accuracy_value, n_images)

    print("loss", loss_tracker.avg)
    print("accuracy", accuracy_tracker.avg)
