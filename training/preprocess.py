def run_preprocessing() :
    # ------ IMPORTS

    import os
    # --- Library imports 

    import pandas as pd 
    import numpy as np
    from skmultilearn.model_selection import iterative_train_test_split

    # --- Getting combined dataset

    df=pd.concat([pd.read_csv("data/raw/full_dataset/goemotions_1.csv"),
              pd.read_csv("data/raw/full_dataset/goemotions_2.csv"),
              pd.read_csv("data/raw/full_dataset/goemotions_3.csv")
             ],ignore_index=True)


    # ------ Data Exploration and Preprocessing

    print("The columns in the dataset are : ")
    print(df.head().T)
    print("\n info : ")
    print("\n", df.info())

    # Unnecessary columns
    drop_cols = [
    'id', 'author', 'subreddit', 'link_id',
    'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'
    ]
    # (y columns)
    emotion_cols = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse','sadness',
    'surprise','neutral'
     ]

    # Droping unecessary columns
    df_model = df.drop(columns=drop_cols)

    # Rows with no emotion at all
    no_label_rows = (df_model[emotion_cols].sum(axis=1) == 0).sum()
    print("Rows with no emotion:", no_label_rows)
    # Droping rows with no emotion
    valid_rows = df_model[emotion_cols].sum(axis=1) > 0
    df_model = df_model[valid_rows].reset_index(drop=True)

    # saving final dataset
    df_model.to_csv(r'data\processed\processed.csv', index=False) 


    # ------ Data Splitting     
                                # Training set   = 70%, 
    X = df_model['text']            # Validation set = 15%, 
    y = df_model[emotion_cols]      # Test set       = 15%

    X_arr = X.to_numpy().reshape(-1, 1)  # shape (num_samples, 1)
    y_arr = y.to_numpy()                # shape (num_samples, num_emotions)

    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X_arr,y_arr,test_size=0.3)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp,y_temp,test_size=0.5)

    X_train = pd.Series(X_train.flatten())
    X_val   = pd.Series(X_val.flatten())
    X_test  = pd.Series(X_test.flatten())

    y_train = pd.DataFrame(y_train, columns=emotion_cols)
    y_val   = pd.DataFrame(y_val, columns=emotion_cols)
    y_test  = pd.DataFrame(y_test, columns=emotion_cols)


    # Print dataset sizes
    print("Train label density mean:", y_train.sum(axis=1).mean())
    print("Val label density mean:", y_val.sum(axis=1).mean())
    print("Test label density mean:", y_test.sum(axis=1).mean())
    print("******************************")
    print(y.sum().sort_values())


    np.save("data/splitting/X_T.npy", X_train.to_numpy())
    np.save("data/splitting/y_train.npy", y_train.to_numpy())

    np.save("data/splitting/X_val.npy", X_val.to_numpy())
    np.save("data/splitting/y_val.npy", y_val.to_numpy())

    np.save("data/splitting/X_test.npy", X_test.to_numpy())
    np.save("data/splitting/y_test.npy", y_test.to_numpy())

    np.save("data/splitting/emotion_labels.npy", np.array(emotion_cols))

    print("✅ All artifacts saved successfully")