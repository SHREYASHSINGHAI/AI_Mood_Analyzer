def run_feature_engineering():
    import numpy as np
    import os
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib


    # ------ Importing freezed datasets ------

    # ---  File Path

    base_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    splits_dir    = os.path.join(base_dir,"data", "splitting")
    artifacts_dir = os.path.join(base_dir, "artifacts")

    # --- Loading Datasets

    X_train = np.load(os.path.join(splits_dir, "X_T.npy"), allow_pickle=True)
    X_test  = np.load(os.path.join(splits_dir, "X_test.npy"), allow_pickle=True)
    X_val   = np.load(os.path.join(splits_dir, "X_val.npy"), allow_pickle=True)


    # ------ Vectorising the text files using TF-IDF Vectorizer ------

    tfidf = TfidfVectorizer(max_features = 15000, # vocabulary size
                            ngram_range = (1,2),  # unigrams and bigrams
                            min_df = 5,           # must appear in at least 5 docs
                            max_df = 0.9,         # must not appear in more than 90% of docs
                            sublinear_tf = True)  # 1+log(count) to adjust word importance


    # ------ Fitting and transforming the training data ------

    # --- Fitting on training data

    X_train_vec = tfidf.fit_transform(X_train)

    # --- Transforming validation and testing data

    X_val_vec = tfidf.transform(X_val)
    X_test_vec = tfidf.transform(X_test)

    print("\n TF-IDF Vectorization completed!")


    # ------ Saving the vectorized features and the TF-IDF model ------


    joblib.dump(tfidf, os.path.join(artifacts_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(X_train_vec, os.path.join(artifacts_dir, "X_train_vec.pkl"))
    joblib.dump(X_val_vec, os.path.join(artifacts_dir, "X_val_vec.pkl"))
    joblib.dump(X_test_vec, os.path.join(artifacts_dir, "X_test_vec.pkl"))

    print("\n✅ All artifacts saved successfully")