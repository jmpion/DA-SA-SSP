# The purpose of this file is to run three experiments.
# 1) With a pre-trained BERT transformer, fine-tune it for sentiment analysis on the Multi-Domain Sentiment Dataset.
# 2) Before fine-tuning the transformer for sentiment analysis, apply a pre-training with a DG-Mix framework. (to be implemented in a future version of this work)
# 3) Before fine-tuning the transformer for sentiment analysis, apply a pre-training with a VICReg framework.
# Considering the amount of resource, time and computation needed to pre-train and fine-tune such a model, I won't spend much time in finding the best parameters.
# Simple experiments should be run to estimate if VICReg or DG-Mix approaches are suitable to pre-train a transformer for sentiment analysis under domain adaptation.

import numpy as np
import pickle
from tqdm import tqdm
from utils import define_logger, get_reviews_and_target, unwrap_BERT_data
from loops import evaluation_BERT
from pre_training_losses import consistency_loss, covariance_loss, variance_loss, triangular_loss
from plyer import notification
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers

import nlpaug.augmenter.word as naw

# For the parser.
import argparse

# PARAMETERS...
# ... for pre-training.
BATCH_SIZE_PRETRAINING = 20
N_EPOCHS_PRETRAINING = 5
LEARNING_RATE_PRETRAINING = 1e-6 # 1e-5 # 1e-6 # 1e-7 # BEST: 1e-6
# ... for the rest.
LEARNING_RATE = 1e-3
BATCH_SIZE = 200
N_EPOCHS = 50
MAX_LENGTH = 256
EXPERIMENT_PATH = "."

class BertDataset(Dataset):
    def __init__(self, reviews, Y, tokenizer, max_length, in_flight=True):
        super(BertDataset, self).__init__()
        self.tokenizer=tokenizer
        self.reviews = reviews
        self.Y = Y
        self.max_length=max_length
        self.in_flight = in_flight
        # I let this option to tokenize in-flight or not in-flight, to choose between space and time complexity.
        if self.in_flight==False:
            self.already_encoded = [
                self.tokenizer.encode_plus(
                    text=text,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    max_length=self.max_length,
                )
                for text in self.reviews
            ]
            del self.reviews
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        if self.in_flight==False:
            inputs = self.already_encoded[index]
        else:
            inputs = self.tokenizer.encode_plus(
                text=self.reviews[index],
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=self.max_length,
            )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.Y[index], dtype=torch.long)
            }

class ConcatDataset(Dataset):
    """This dataset is useful to concatenate two datasets that need the same order, for VICReg."""
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = len(self.dataset1)
    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]
    def __len__(self):
        return self.length

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)
        
    def forward(self, ids, mask, token_type_ids, pretraining=False):
        _, embedding = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        if pretraining:
            return embedding
        else:
            out= self.out(embedding)
            return out

class VICReg_head(nn.Module):
    def __init__(self):
        super(VICReg_head, self).__init__()
        self.linear1 = nn.Linear(768, 1000)
    
    def forward(self, x):
        return self.linear1(x)

if __name__=="__main__":
    # Parsing arguments.
    parser = argparse.ArgumentParser(description="BERT experiment.")
    parser.add_argument("--experiment_mode", default="normal", help="normal for BERT without custom pretraining. vicreg for BERT with VICReg pretraining. dgmix for VICReg with DG-Mix pretraining.")
    parser.add_argument("--load_model", default="none", help="none to train a brand new model. model weights to load a model from its save weights.")
    args = parser.parse_args()
    EXPERIMENT_MODE = args.experiment_mode
    LOAD_MODEL = args.load_model

    # Defining my logger to save and show the experiment progress.
    logger_name = "BERT_logger"
    logger_output_file = f"{EXPERIMENT_PATH}/log_BERT_{EXPERIMENT_MODE}.txt"
    logger = define_logger(logger_name, logger_output_file)
    try:
        # Selecting the GPU.
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEVICE USED : {DEVICE}")

        # Let's start the experiment.
        product_type_list = ["books", "dvd", "electronics", "kitchen_&_housewares"]
        pairs_already_seen = [
                                # ("books", "dvd"),
                                # ("books", "electronics"),
                                # ("books", "kitchen_&_housewares"),
                                # ("dvd", "books"),
                                # ("dvd", "electronics"),
                                # ("dvd", "kitchen_&_housewares"),
                                # ("electronics", "books"),
                                # ("electronics", "dvd"),
                                # ("electronics", "kitchen_&_housewares"),
                                # ("kitchen_&_housewares", "books"),
                                # ("kitchen_&_housewares", "dvd"),
                            ]
        for product_source in product_type_list:
            for product_target in product_type_list:
                results = {
                    "training losses" : [],
                    "training accuracies" : [],
                    "test losses" : [],
                    "test accuracies" : [],
                }
                if product_target != product_source and (product_source, product_target) not in pairs_already_seen:
                    logger.info(f"#### BERT EXPERIMENT ####\n* SOURCE : {product_source}\n* TARGET : {product_target}\n")
                    # Loading the data.
                    reviews_source, Y_source = get_reviews_and_target(product_source)
                    reviews_target, Y_target = get_reviews_and_target(product_target)
                    # Tokenizer.
                    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
                    # Instantiating the model.
                    model = BERT().to(DEVICE)
                    # Trying to load a model from files.
                    if LOAD_MODEL != "none":
                        try:
                            weights_path = f"{LOAD_MODEL}_SOURCE_{product_source}_TARGET_{product_target}_{EXPERIMENT_MODE}_weights.pth"
                            model.load_state_dict(torch.load(weights_path))
                        except Exception as e:
                            logger.info("Model not loaded successfully. Will initialize a new model. Reason below:")
                            logger.info(e)

                    # PRE-TRAINING PHASE
                    for params in model.parameters():
                        params.requires_grad = True
                    model.train()
                    if EXPERIMENT_MODE in ["vicreg", "dgmix"]:
                        if EXPERIMENT_MODE == "vicreg":
                            pretrainer = VICReg_head().to(DEVICE)
                        elif EXPERIMENT_MODE == "dgmix":
                            logger.info("DG-Mix is currently not supported. The results you will get are to be erroneous.")
                        # First, as we do unsupervised pre-training, we can group the source and the target together.
                        reviews_all = np.concatenate([reviews_source, reviews_target], axis=0)
                        Y_all = np.concatenate([Y_source, Y_target], axis=0)
                        # Dataset for source + target.
                        dataset_all = BertDataset(reviews_all, Y_all, tokenizer, max_length=MAX_LENGTH)
                        # Dataloader for source + target.
                        # Second dataset for VICReg approach.
                        if EXPERIMENT_MODE == "vicreg":
                            aug = naw.SynonymAug()
                            second_reviews_all = np.array(aug.augment(list(reviews_all)))
                            del reviews_all
                            second_dataset_all = BertDataset(second_reviews_all, Y_all, tokenizer, max_length=MAX_LENGTH)
                            del second_reviews_all
                            concatenated_dataset = ConcatDataset(dataset_all, second_dataset_all)
                            del dataset_all
                            del second_dataset_all
                            dataloader_all = DataLoader(concatenated_dataset, batch_size=BATCH_SIZE_PRETRAINING, shuffle=True)
                        elif EXPERIMENT_MODE == "dgmix":
                            second_dataset_all = BertDataset(reviews_all, Y_all, tokenizer, max_length=MAX_LENGTH)
                            del reviews_all
                            dataloader_all = zip(
                                DataLoader(dataset_all, batch_size=BATCH_SIZE_PRETRAINING, shuffle=True),
                                DataLoader(dataset_all, batch_size=BATCH_SIZE_PRETRAINING, shuffle=True)
                            )
                            del dataset_all
                        # Optimizer.
                        optimizer = torch.optim.Adam(params=list(model.parameters()) + list(pretrainer.parameters()), lr=LEARNING_RATE_PRETRAINING)
                        # Pre-training loop.
                        for epoch in range(N_EPOCHS_PRETRAINING):
                            # Tracing the losses in my experiments.
                            if EXPERIMENT_MODE == "vicreg":
                                current_consistency_loss = 0
                            elif EXPERIMENT_MODE == "dgmix":
                                current_triangular_loss = 0
                            current_covariance_loss = 0
                            current_variance_loss = 0
                            current_total_loss = 0
                            # Number of batches.
                            n_batches = len(dataloader_all)
                            # Iterating through the dataset.
                            for data1, data2 in tqdm(dataloader_all):
                                # Get all data and send to GPU.
                                ids1, token_type_ids1, mask1, Y1 = unwrap_BERT_data(data1, DEVICE)
                                del data1
                                ids2, token_type_ids2, mask2, Y2 = unwrap_BERT_data(data2, DEVICE)
                                del data2

                                # Lambda coefficient for DG-Mix.
                                if EXPERIMENT_MODE == "dgmix":
                                    alpha = 0.2
                                    beta = 0.2
                                    lamda = np.random.beta(alpha, beta)

                                # Forward pass.
                                optimizer.zero_grad()
                                E1 = model(ids=ids1, mask=mask1, token_type_ids=token_type_ids1, pretraining=True)
                                E2 = model(ids=ids2, mask=mask2, token_type_ids=token_type_ids2, pretraining=True)
                                Z1 = pretrainer(E1)
                                Z2 = pretrainer(E2)

                                # Losses.
                                loss_consistency = consistency_loss(Z1, Z2)
                                loss_covariance = covariance_loss(Z1, Z2)
                                loss_variance = variance_loss(Z1, Z2)
                                total_loss = 25 * loss_consistency + 25 * loss_variance + loss_covariance

                                # Backpropagation.
                                total_loss.backward()
                                optimizer.step()

                                # Tracing losses.
                                current_consistency_loss += loss_consistency.item()
                                current_covariance_loss += loss_covariance.item()
                                current_variance_loss += loss_variance.item()
                                current_total_loss += total_loss.item()
                            logger.info(f"Epoch : {epoch + 1}/{N_EPOCHS_PRETRAINING} ; Total Loss : {current_total_loss / n_batches:.3f}")
                            logger.info(f"Consistency Loss : {current_consistency_loss / n_batches:.3f}")
                            logger.info(f"Covariance Loss : {current_covariance_loss / n_batches:.3f}")
                            logger.info(f"Variance Loss : {current_variance_loss / n_batches:.3f}")
                        del dataloader_all
                    # Making the datasets for the training and test phases.
                    dataset_source = BertDataset(reviews_source, Y_source, tokenizer, max_length=MAX_LENGTH)
                    dataset_target = BertDataset(reviews_target, Y_target, tokenizer, max_length=MAX_LENGTH)
                    # TRAINING PHASE
                    # Loss + Optimizer.
                    loss_fn = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
                    # Dataloader.
                    dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True)
                    # Training iterations.
                    # For pure training, once the transformer is pre-trained, we don't need to update its parameters.
                    for param in model.bert_model.parameters():
                        param.requires_grad = False
                    for epoch in range(N_EPOCHS):
                        model.train()
                        # Loss for one epoch.
                        current_loss = 0
                        # To trace the performance on the source dataset at each epoch.
                        total_num_correct = 0
                        total_num_samples = 0
                        # Number of batches.
                        n_batches = len(dataloader_source)
                        # Iterating throught the data.
                        for data in dataloader_source:
                            # Get all data and send to GPU.
                            ids = data['ids'].to(DEVICE)
                            token_type_ids = data['token_type_ids'].to(DEVICE)
                            mask = data['mask'].to(DEVICE)
                            Y = data['target'].to(DEVICE)
                            Y = Y.unsqueeze(1)

                            # Emptying memory a bit.
                            del data

                            # Forward pass.
                            optimizer.zero_grad()
                            output = model(
                                ids=ids,
                                mask=mask,
                                token_type_ids=token_type_ids)
                            Y = Y.type_as(output)

                            # Backpropagation
                            loss = loss_fn(output, Y)
                            loss.backward()
                            optimizer.step()

                            # Accumulating loss
                            current_loss += loss.item()

                            # Comparing the predictions to the target.
                            pred = np.where(output.cpu() >= 0, 1, 0)
                            num_correct = sum(1 for a, b in zip(pred, Y) if a[0] == b[0])
                            num_samples = pred.shape[0]
                            total_num_correct += num_correct
                            total_num_samples += num_samples

                        # Tracing the results during the training.
                        loss_training = current_loss / n_batches
                        accuracy_training = 100 * total_num_correct / total_num_samples
                        logger.info(f"Epoch : {epoch + 1}/{N_EPOCHS} ; Loss : {loss_training:.3f} ; Accuracy : {accuracy_training:.2f}%")
                        
                        # Evaluation phase at each epoch.
                        loss_test, accuracy_test = evaluation_BERT(model, dataset_target, loss_fn, BATCH_SIZE, DEVICE, logger)

                        # Saving accuracies and losses.
                        results["training losses"].append(loss_training)
                        results["training accuracies"].append(accuracy_training)
                        results["test losses"].append(loss_test)
                        results["test accuracies"].append(accuracy_test)

                    # Saving the model once the training is done.
                    torch.save(model.state_dict(), f"BERT_model_SOURCE_{product_source}_TARGET_{product_target}_{EXPERIMENT_MODE}_weights.pth")
                    
                    # Plots.
                    time_range = np.arange(1, N_EPOCHS + 1)
                    # Losses.
                    plt.plot(time_range, results["training losses"], color="blue", label="Training loss")
                    plt.plot(time_range, results["test losses"], color="red", label="Test loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(f"BERT_{EXPERIMENT_MODE}_{product_source}_{product_target}_losses.jpg")
                    plt.show()
                    # Accuracies.
                    plt.plot(time_range, results["training accuracies"], color="blue", label="Training accuracy")
                    plt.plot(time_range, results["test accuracies"], color="red", label="Test accuracy")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.legend()
                    plt.savefig(f"BERT_{EXPERIMENT_MODE}_{product_source}_{product_target}_accuracies.jpg")
                    plt.show()

                    # Save values.
                    with open(f"saved_results_BERT_{EXPERIMENT_MODE}_{product_source}_{product_target}.pkl", "wb") as f:
                        pickle.dump(results, f)
        notification.notify(title="Finished running script!", message="Successful.")
    except Exception as e:
        logger.info(e)
        notification.notify(title="Finished running script!", message="Fail.")