from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import unwrap_BERT_data
import numpy as np

def evaluation_NN(model, dataset_target, loss_fn, device, logger):
    # Evaluation phase.
    dataloader_target = DataLoader(dataset_target, batch_size=len(dataset_target), shuffle=False)
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader_target:
            X, Y = X.to(device), Y.to(device)
            pred_test = model(X)
        loss_test = loss_fn(pred_test, Y).item()
        # Outputting the accuracy.
        Y, pred_test = Y.cpu(), pred_test.cpu()
        accuracy_test = 100 * np.mean(np.argmax(Y.numpy(), axis=1) == np.argmax(pred_test.numpy(), axis=1))
    logger.info(f"Target loss : {loss_test:.3f} ; Accuracy : {accuracy_test:.2f}%\n")
    return loss_test, accuracy_test

def evaluation_BERT(model, dataset_target, loss_fn, batch_size, device, logger):
    # Evaluation phase.
    dataloader_target_test = DataLoader(dataset_target, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        total_loss_test = 0
        total_correct = 0
        total_samples = 0
        n_batches = len(dataloader_target_test)
        for data in tqdm(dataloader_target_test):
            # Getting the data and sending it to the GPU.
            ids, token_type_ids, mask, Y = unwrap_BERT_data(data, device)

            # Emptying the memory a bit.
            del data

            # Forward pass.
            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            Y = Y.type_as(output)

            # Getting the results...
            # ... for the loss.
            loss_test = loss_fn(output, Y).item()
            total_loss_test += loss_test
            #... for the accuracy.
            Y, output = Y.cpu().numpy(), output.cpu().numpy()
            pred = np.where(output >= 0, 1, 0)
            total_correct += np.sum(Y == pred)
            total_samples += len(Y)
        # Logging the results.
        loss_test = total_loss_test / n_batches
        accuracy = 100 * total_correct / total_samples
        logger.info(f"Target loss : {loss_test:.3f} ;  Target Accuracy : {accuracy:.2f}%")
        return loss_test, accuracy