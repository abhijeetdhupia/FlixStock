import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score

from model import OutputModel
from dataset import FashionDataset, DatasetAttributes, mean, std


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_neck = 0
        accuracy_sleeve = 0
        accuracy_pattern = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_neck, batch_accuracy_sleeve, batch_accuracy_pattern = \
                calculate_metrics(output, target_labels)

            accuracy_neck += batch_accuracy_neck
            accuracy_sleeve += batch_accuracy_sleeve
            accuracy_pattern += batch_accuracy_pattern

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_neck /= n_samples
    accuracy_sleeve /= n_samples
    accuracy_pattern /= n_samples
    print('-' * 75)
    print("Validation  loss: {:.4f}, neck: {:.4f}, sleeve: {:.4f}, pattern: {:.4f}\n".format(
        avg_loss, accuracy_neck, accuracy_sleeve, accuracy_pattern))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_neck', accuracy_neck, iteration)
    logger.add_scalar('val_accuracy_sleeve', accuracy_sleeve, iteration)
    logger.add_scalar('val_accuracy_pattern', accuracy_pattern, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_neck_all = []
    gt_sleeve_all = []
    gt_pattern_all = []
    predicted_neck_all = []
    predicted_sleeve_all = []
    predicted_pattern_all = []

    accuracy_neck = 0
    accuracy_sleeve = 0
    accuracy_pattern = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_necks = batch['labels']['neck_labels']
            gt_sleeves = batch['labels']['sleeve_labels']
            gt_patterns = batch['labels']['pattern_labels']
            output = model(img.to(device))

            batch_accuracy_neck, batch_accuracy_sleeve, batch_accuracy_pattern = \
                calculate_metrics(output, batch['labels'])
            accuracy_neck += batch_accuracy_neck
            accuracy_sleeve += batch_accuracy_sleeve
            accuracy_pattern += batch_accuracy_pattern

            # get the most confident prediction for each image
            _, predicted_necks = output['neck'].cpu().max(1)
            _, predicted_sleeves = output['sleeve'].cpu().max(1)
            _, predicted_patterns = output['pattern'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_neck = attributes.neck_id_to_name[predicted_necks[i].item()]
                predicted_sleeve = attributes.sleeve_id_to_name[predicted_sleeves[i].item()]
                predicted_pattern = attributes.pattern_id_to_name[predicted_patterns[i].item()]

                gt_neck = attributes.neck_id_to_name[gt_necks[i].item()]
                gt_sleeve = attributes.sleeve_id_to_name[gt_sleeves[i].item()]
                gt_pattern = attributes.pattern_id_to_name[gt_patterns[i].item()]

                gt_neck_all.append(gt_neck)
                gt_sleeve_all.append(gt_sleeve)
                gt_pattern_all.append(gt_pattern)

                predicted_neck_all.append(predicted_neck)
                predicted_sleeve_all.append(predicted_sleeve)
                predicted_pattern_all.append(predicted_pattern)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_sleeve, predicted_pattern, predicted_neck))
                gt_labels.append("{}\n{}\n{}".format(gt_sleeve, gt_pattern, gt_neck))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nneck: {:.4f}, sleeve_length: {:.4f}, pattern: {:.4f}".format(
            accuracy_neck / n_samples,
            accuracy_sleeve / n_samples,
            accuracy_pattern / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # neck
        cn_matrix = confusion_matrix(
            y_true=gt_neck_all,
            y_pred=predicted_neck_all,
            labels=attributes.neck_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.neck_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("necks")
        plt.savefig("cf_neck.png")
        plt.tight_layout()
        plt.show()

        # sleeve
        cn_matrix = confusion_matrix(
            y_true=gt_sleeve_all,
            y_pred=predicted_sleeve_all,
            labels=attributes.sleeve_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.sleeve_labels).plot(
            xticks_rotation='horizontal')
        plt.title("sleeves")
        plt.savefig("cf_sleeves.png")
        plt.tight_layout()
        plt.show()

        # Uncomment code below to see the pattern confusion matrix (it may be too big to display)
        cn_matrix = confusion_matrix(
            y_true=gt_pattern_all,
            y_pred=predicted_pattern_all,
            labels=attributes.pattern_labels,
            normalize='true')
        plt.rcParams.update({'font.size': 1.8})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(cn_matrix, attributes.pattern_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("Pattern Types")
        plt.savefig("cf_pattern.png")
        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_neck = output['neck'].cpu().max(1)
    gt_neck = target['neck_labels'].cpu()

    _, predicted_sleeve = output['sleeve'].cpu().max(1)
    gt_sleeve = target['sleeve_labels'].cpu()

    _, predicted_pattern = output['pattern'].cpu().max(1)
    gt_pattern = target['pattern_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_neck = balanced_accuracy_score(y_true=gt_neck.numpy(), y_pred=predicted_neck.numpy())
        accuracy_sleeve = balanced_accuracy_score(y_true=gt_sleeve.numpy(), y_pred=predicted_sleeve.numpy())
        accuracy_pattern = balanced_accuracy_score(y_true=gt_pattern.numpy(), y_pred=predicted_pattern.numpy())

    return accuracy_neck, accuracy_sleeve, accuracy_pattern


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="./checkpoints/2021-06-27_20-45/checkpoint-000100.pth")
    parser.add_argument('--attributes_file', type=str, default='./final_attributes.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = DatasetAttributes(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = OutputModel(n_neck_classes=attributes.num_necks, n_sleeve_classes=attributes.num_sleeves,
                             n_pattern_classes=attributes.num_patterns).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)