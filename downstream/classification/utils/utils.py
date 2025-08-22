import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def validate(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    # Disable gradient computation for validation.
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            # Move images and labels to the specified device.
            images = batch_data["img_w"].to(device)
            labels = batch_data["target"].to(device)
            outputs = model(images)

            # Obtain predictions by selecting the index with the maximum logit.
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # Calculate probabilities using softmax.
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    # Concatenate predictions and labels from all batches.
    predicts = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.cat(all_probs, dim=0)

    # Calculate accuracy.
    accuracy = (torch.sum(predicts == labels).item()/len(labels))

    # Calculate AUC.
    probs = torch.cat(all_probs, dim=0)
    AUC = roc_auc_score(labels, probs[:,1])

    return accuracy, AUC
