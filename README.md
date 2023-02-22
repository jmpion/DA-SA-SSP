# DA-SA-SSP
Domain Adaptation for Sentiment Analysis using Self-Supervised Pre-training. (It explains the name of the repository DA-SA-SSP)

- `sorted_data_acl.zip` and `domain_sentiment_data.tar.gz` are too big to be uploaded to the repository, so I provide access links here, for the moment: 
  - https://drive.google.com/file/d/1fuOG-wkYx8N_d66v-1FrmcbA7wWgVcqe/view?usp=sharing
  - https://drive.google.com/file/d/1OwDQQxvFY_2Pmae68zn4VsidgOvTRC5z/view?usp=sharing

The `sorted_data_acl.zip` archive should be extracted before doing experiments.

Requirements can be installed with the `requirements.txt` file.

If the files of the repository are put in the data folder of a Colab session, it is possible to run the VICReg experiments by running

`!unzip sorted_data_acl.zip`

`!pip install transformers`

`!pip install plyer`

`!pip install nlpaug`

`!python experiments_BERT.py --experiment_mode vicreg`

Note that `sorted_data_acl.zip` contains the dataset after I preprocessed it.

Files that are only necessary before generating this preprocessed dataset are given in the repository but they can be overlooked if you only want to run the experiments. They are the following:

- `convert_dataset_to_csv.py`
- `domain_sentiment_data.tar.gz`

The repository is not a final version of this work. For example, random seeds should be added to ensure complete reproducibility of the experiments.
