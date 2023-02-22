import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import sys

def define_logger(logger_name, logger_output_file):
    """Defines a logger to output and save results.
    logger_output_file is the place (path+specific file where the logs are going to be saved.
    NB: This function overwrites previous logs if the logging file already existed."""
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(filename=logger_output_file, mode="w")
    file_handler.setLevel(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

def get_dataframe(product_type):
    """ Get the dataframe of all negative and positive reviews for product_type"""
    df_negative = pd.read_csv(f"sorted_data_acl/{product_type}/negative.csv")
    df_negative["sentiment"] = 0
    df_positive = pd.read_csv(f"sorted_data_acl/{product_type}/positive.csv")
    df_positive["sentiment"] = 1
    df_all = pd.concat([df_negative, df_positive], axis=0).reset_index(drop=True)
    return df_all

def get_reviews_and_target(product_type):
    """Get all the reviews and the target for a given product type."""
    df = get_dataframe(product_type)
    reviews = df["review_text"].values
    Y = df["sentiment"].values
    return reviews, Y

def rcv_svm(X_source, Y_source, X_target, Y_target):
    """ Returns the C parameter obtained by reverse cross-validation."""
    C_range = np.logspace(-5, -3, num=10)
    accuracies_grid = np.zeros(shape=(10, 10))
    for i, C in tqdm(enumerate(C_range)):
        skf_source = StratifiedKFold(n_splits=10).split(X_source, Y_source)
        skf_target = StratifiedKFold(n_splits=10).split(X_target, Y_target)
        for j, ((train_index_source, test_index_source), (train_index_target, test_index_target)) in enumerate(zip(skf_source, skf_target)):
            X_source_train = X_source[train_index_source, :]
            Y_source_train = Y_source[train_index_source]

            model_1 = svm.LinearSVC(C=C)
            model_1.fit(X_source_train, Y_source_train)

            X_reverse = X_target[train_index_target, :]
            Y_reverse = model_1.predict(X_reverse)

            model_2 = svm.LinearSVC(C=C)
            model_2.fit(X_reverse, Y_reverse)

            Y_validation = Y_source[test_index_source]
            X_validation = X_source[test_index_source, :]
            Y_pred_validation = model_2.predict(X_validation)

            accuracy = accuracy_score(Y_validation, Y_pred_validation)
            accuracies_grid[i, j] = accuracy
    mean_accuracies = np.mean(accuracies_grid, axis=1)
    best_idx = np.argmax(mean_accuracies, axis=0)
    best_C = C_range[best_idx]
    return best_C
        
def features_extraction(reviews_source, reviews_target, mode="crossed_tfidf"):
    if mode=="count":
        vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        vectorizer = vectorizer.fit(reviews_source)

        X_source = vectorizer.transform(reviews_source).toarray()
        X_target = vectorizer.transform(reviews_target).toarray()

        return X_source, X_target, vectorizer

    elif mode=="tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        vectorizer = vectorizer.fit(reviews_source)

        X_source = vectorizer.transform(reviews_source).toarray()
        X_target = vectorizer.transform(reviews_target).toarray()

        return X_source, X_target, vectorizer

    elif "crossed" in mode:
        vectorizer_source = CountVectorizer(max_features=10000, ngram_range=(1, 2), binary=True)
        vectorizer_source = vectorizer_source.fit(reviews_source)

        vectorizer_target = CountVectorizer(max_features=10000, ngram_range=(1, 2), binary=True)
        vectorizer_target = vectorizer_target.fit(reviews_target)

        X_source = vectorizer_source.transform(reviews_source).toarray()
        X_target = vectorizer_target.transform(reviews_target).toarray()

        counts_source = np.sum(X_source, axis=0)
        counts_target = np.sum(X_target, axis=0)
        global_counts = {"source" : counts_source, "target" : counts_target}

        global_hash = {"source" : {}, "target" : {}}

        vectorizers = [vectorizer_source, vectorizer_target]
        domains = ["source", "target"]
        for vectorizer, domain in zip(vectorizers, domains):
            other_domain = "source" if domain == "target" else "target"
            for word in vectorizer.vocabulary_:
                global_hash[domain][word] = global_counts[domain][vectorizer.vocabulary_[word]]
                if word not in global_hash[other_domain]:
                    global_hash[other_domain][word] = 0

        for domain in domains:
            global_hash[domain] = {word : global_hash[domain][word] for word in sorted(global_hash[domain])}

        idx_to_word = {i : word for i, word in enumerate(global_hash["source"])}
        all_counts_source = np.array([global_hash["source"][word] for word in global_hash["source"]])
        all_counts_target = np.array([global_hash["target"][word] for word in global_hash["target"]])

        all_min_counts = np.minimum(all_counts_source, all_counts_target)
        idx_best = (all_min_counts.argsort())[::-1]

        crossed_vocabulary = {idx_to_word[idx] : i for i, idx in enumerate(idx_best[:5000])}

        if mode=="crossed_count":
            crossed_vectorizer = CountVectorizer(vocabulary=crossed_vocabulary, ngram_range=(1, 2))
        elif mode=="crossed_tfidf":
            crossed_vectorizer = TfidfVectorizer(vocabulary=crossed_vocabulary, ngram_range=(1, 2))
            crossed_vectorizer = crossed_vectorizer.fit(np.concatenate([reviews_source, reviews_target], axis=0))

        X_source_crossed = crossed_vectorizer.transform(reviews_source).toarray()
        X_target_crossed = crossed_vectorizer.transform(reviews_target).toarray()
        return X_source_crossed, X_target_crossed, crossed_vectorizer

def unwrap_BERT_data(data, device):
    ids = data['ids'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    mask = data['mask'].to(device)
    Y = data['target'].to(device)
    Y = Y.unsqueeze(1)
    return ids, token_type_ids, mask, Y