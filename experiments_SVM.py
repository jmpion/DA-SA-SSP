import logging
import numpy as np
from utils import get_dataframe, features_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from utils import rcv_svm
from plyer import notification

# For the parser.
import argparse

parser = argparse.ArgumentParser(description="SVM experiment.")
parser.add_argument("--vectorizer", default="count", help="Choose CountVectorizer or TfidfVectorizer, or CrossedVectorizer. count, tfidf, crossed_count or crossed_tfidf")
args = parser.parse_args()

EXPERIMENT_PATH = "logs/1_CrossedVectorizer_Experiments"

if __name__=="__main__":
    try:
        # Defining my logger to save and show the experiment progress.
        import sys
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level=logging.INFO)
        file_handler = logging.FileHandler(filename=f"{EXPERIMENT_PATH}/log_SVM_{args.vectorizer}Vectorizer.txt", mode="w")
        file_handler.setLevel(level=logging.INFO)
        logger = logging.getLogger("SVM_logger")
        logger.setLevel(level=logging.INFO)
        logger.addHandler(console)
        logger.addHandler(file_handler)

        # Let's start the experiment.
        product_type_list = ["books", "dvd", "electronics", "kitchen_&_housewares"]
        for product_source in product_type_list:
            for product_target in product_type_list:
                if product_target != product_source:
                    logger.info(f"#### EXPERIMENT ####\n* SOURCE : {product_source}\n* TARGET : {product_target}\n")
                    # Loading the data and making the datasets.
                    df_source = get_dataframe(product_source)
                    df_target = get_dataframe(product_target)

                    reviews_source = df_source["review_text"].values
                    Y_source = df_source["sentiment"].values

                    reviews_target = df_target["review_text"].values
                    Y_target = df_target["sentiment"].values

                    X_source, X_target, _ = features_extraction(reviews_source, reviews_target, mode=args.vectorizer)

                    # Instantiating the model.
                    C = rcv_svm(X_source, Y_source, X_target, Y_target)
                    model = svm.LinearSVC(C=C)

                    # Training phase.
                    model.fit(X_source, Y_source)

                    # Evaluation phase.
                    pred_test = model.predict(X_target)

                    # Outputting the accuracy.
                    accuracy = accuracy_score(Y_target, pred_test)
                    logger.info(f"Accuracy : {100 * accuracy:.2f}% with...\n* SOURCE : {product_source}\n* TARGET : {product_target}\n")
        notification.notify(title="Finished running script!", message="Successful.")
    except:
        notification.notify(title="Finished running script!", message="Fail.")