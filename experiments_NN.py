import logging
import numpy as np
from utils import get_reviews_and_target, features_extraction
from plyer import notification
from loops import evaluation_NN
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For the parser.
import argparse

# PARAMETERS.
LEARNING_RATE = 1e-3
BATCH_SIZE = 200
N_EPOCHS = 200

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).float()
        y_label = self.Y[idx]
        y = torch.zeros(2)
        y[y_label] = 1
        return x, y

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        self.linear1 = nn.Linear(5000, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, 2)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.linear2(x)
        return x

if __name__=="__main__":
    try:
        # Parsing arguments.
        parser = argparse.ArgumentParser(description="Neural Network experiment.")
        parser.add_argument("--vectorizer", default="count", help="Choose CountVectorizer or TfidfVectorizer, or CrossedVectorizer. count, tfidf, crossed_count or crossed_tfidf")
        parser.add_argument("--experiment", default="vectorizer", help="vectorizer or ...")
        parser.add_argument("--neurons", default=50, help="Number of neurons in the hidden layer.")
        args = parser.parse_args()

        HIDDEN_DIM = int(args.neurons)
        if args.experiment == "vectorizer":
            EXPERIMENT_PATH = "logs/1_CrossedVectorizer_Experiments"

        # Defining my logger to save and show the experiment progress.
        import sys
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level=logging.INFO)
        file_handler = logging.FileHandler(filename=f"{EXPERIMENT_PATH}/log_NN_{HIDDEN_DIM}-neurons_{args.vectorizer}Vectorizer.txt", mode="w")
        file_handler.setLevel(level=logging.INFO)
        logger = logging.getLogger("NN_logger")
        logger.setLevel(level=logging.INFO)
        logger.addHandler(console)
        logger.addHandler(file_handler)

        # Selecting the GPU.
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEVICE USED : {DEVICE}")

        # Let's start the experiment.
        product_type_list = ["books", "dvd", "electronics", "kitchen_&_housewares"]
        for product_source in product_type_list:
            for product_target in product_type_list:
                results = {
                    "training losses" : [],
                    "training accuracies" : [],
                    "test losses" : [],
                    "test accuracies" : [],
                }
                if product_target != product_source:
                    logger.info(f"#### EXPERIMENT ####\n* SOURCE : {product_source}\n* TARGET : {product_target}\n")
                    # Loading the data and making the datasets.
                    reviews_source, Y_source = get_reviews_and_target(product_source)
                    reviews_target, Y_target = get_reviews_and_target(product_target)

                    X_source, X_target, _ = features_extraction(reviews_source, reviews_target, mode=args.vectorizer)

                    dataset_source = CustomDataset(X_source, Y_source)
                    dataset_target = CustomDataset(X_target, Y_target)

                    # Instantiating the model.
                    model = SimpleNetwork().to(DEVICE)

                    # Training phase.
                    # Loss + Optimizer.
                    loss_fn = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
                    # Dataloader.
                    dataloader_source = DataLoader(dataset_source, batch_size=BATCH_SIZE, shuffle=True)
                    # Training iterations.
                    model.train()
                    for epoch in range(N_EPOCHS):
                        current_loss = 0
                        n_batches = len(dataloader_source)
                        total_correct = 0
                        total_samples = 0
                        for X, Y in dataloader_source:
                            # Send to GPU.
                            X, Y = X.to(DEVICE), Y.to(DEVICE)

                            # Forward pass
                            pred = model(X)
                            loss = loss_fn(pred, Y)

                            # Backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # Accumulating loss
                            current_loss += loss.item()

                            # Training accuracy
                            total_correct += np.sum(np.argmax(Y.cpu().detach().numpy(), axis=1) == np.argmax(pred.cpu().detach().numpy(), axis=1))
                            total_samples += len(Y)
                        training_accuracy = 100 * total_correct / total_samples
                        training_loss = current_loss / n_batches
                        logger.info(f"Epoch : {epoch}/{N_EPOCHS} ; Training Loss : {training_loss:.3f} ; training Accuracy : {training_accuracy:.2f}%")
                        results["training losses"].append(training_loss)
                        results["training accuracies"].append(training_accuracy)

                        # Evaluation phase.
                        test_loss, test_accuracy = evaluation_NN(model, dataset_target, loss_fn, DEVICE, logger)
                        results["test losses"].append(test_loss)
                        results["test accuracies"].append(test_accuracy)
                    # Plots.
                    time_range = np.arange(1, N_EPOCHS + 1)
                    # Losses.
                    plt.plot(time_range, results["training losses"], color="blue", label="Training loss")
                    plt.plot(time_range, results["test losses"], color="red", label="Test loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(f"NN__{product_source}_{product_target}_losses.jpg")
                    plt.show()
                    # Accuracies.
                    plt.plot(time_range, results["training accuracies"], color="blue", label="Training accuracy")
                    plt.plot(time_range, results["test accuracies"], color="red", label="Test accuracy")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.legend()
                    plt.savefig(f"NN_{HIDDEN_DIM}-neurons_{args.vectorizer}Vectorizer_{product_source}_{product_target}_accuracies.jpg")
                    plt.show()

                    # Save values.
                    with open(f"saved_results_NN_{HIDDEN_DIM}-neurons_{args.vectorizer}Vectorizer_{product_source}_{product_target}.pkl", "wb") as f:
                        pickle.dump(results, f)
        notification.notify(title="Finished running script!", message="Successful.")
    except:
        notification.notify(title="Finished running script!", message="Fail.")