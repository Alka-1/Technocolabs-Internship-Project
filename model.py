# Import important libraries.
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

"""##Load Data"""

track_df = pd.read_csv('tf_mini.csv').set_index('track_id')
log_df = pd.read_csv('log_mini.csv')

"""## Data Preparation"""

# Since there is only 1 feature with 'object' as data type,
# Let's handle it so that we can use all the track features

track_df = pd.get_dummies(data=track_df, columns=['mode'])

track_df.shape, log_df.shape

# Perform left join on the track_id
data = log_df.join(track_df, on='track_id_clean', how='left')

# Remove 'Date' column, as it is not that important for this problem.
data.drop(columns=['date'], inplace=True)



def stack_sessions(data):
    """
    Turn matrix representation into vector by stacking the listen events together (as columns)
    For example:
    session_id session_position feature1 feature2
    a          1                ~        ~
    a          2                ~        ~
    b          1                ~        ~
    b          2                ~        ~
    c          2                ~        ~

    Turns into:

                  feature1              feature2
    session_id   1          2          1          2
    a            ~          ~          ~          ~
    b            ~          ~          ~          ~
    c            Nan        Nan        ~          ~
    """
    value_cols = list(data.columns)
    value_cols.remove('session_id')
    value_cols.remove('session_position')

    unique_sessions = data.pivot(index='session_id', columns='session_position', values=value_cols)
    return unique_sessions


# Columns to be dropped from second half, as it will not be able for prediction.
# We will only use session related information like SessionID, TrackID, SessionPosition for prediction.

drop_cols = list(df.columns)
drop_cols.remove('session_id')
drop_cols.remove('session_position')
drop_cols.remove('track_id_clean')

# Split each train, validation dataset into corresponding (X, y).
# First half of session is stacked and joined to each song in the second half of the session.
#
# Each session is of 20 tracks indicated by SPOS, so we will divide the dataset into two halves, wrt SPOS.
# first-half will contain [1 to 10] tracks data, and second-half will contain remaining [11 to 20] tracks data.
#
#

def split_df(data):
    # Get all the data with SPOS <= 10 ; reset the index: SessionID added again as column ; drop the session_length
    first_half = data.loc[data['session_position'] * 2 <= data['session_length']].reset_index().drop(
        columns=['session_length'])

    # Get all the data with SPOS > 10 ; reset the index: SessionID added again as column
    second_half = data.loc[data['session_position'] * 2 > data['session_length']].reset_index()

    # Drop columns that will be not available for prediction.
    second_half.drop(columns=drop_cols, inplace=True)

    # Stacking the first-half
    first_stacked = stack_sessions(first_half)

    data = second_half.join(first_stacked, how='left', on='session_id')
    return data


def transform(data):
    """**Separate categorical and non-categorical data**"""

    # All categorical data
    data_cat = data.select_dtypes(exclude=['int64', 'float64', 'uint8'])

    # Merge session_position, session_length
    data_cat = pd.merge(data[['session_position', 'session_length']], data_cat, left_index=True, right_index=True)

    # All non-categorical data
    data_non_cat = data.select_dtypes(exclude=['bool', 'object'])
    """## Feature Engineering

    ## PCA

    Use non-categorical data only for PCA.
    """

    # Let's define a Pipeline for data scaling and PCA

    pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=0.3))])

    # Perfrom Standard scaling and Principal component analysis on the non-categorical data

    pca_array = pipe.fit_transform(data_non_cat)
    pca_array.shape  # result is numpy array

    # Convert numpy array to dataframe with the 3 Principal components as columns.
    pc_names = ['PC' + str(i + 1) for i in range(0, pca_array.shape[1])]
    pca_data = pd.DataFrame(pca_array, columns=pc_names)

    """The above 3 principal components are capable of representing 33% of the data (non-categorical)."""

    # Let's re-join this new data of 3 PC's with the categorical data.

    df = pd.merge(data_cat, pca_data, left_index=True, right_index=True)

    # Preprocessed data has only 16 columns now.

    # Convert all data types into numerical data.
    #

    # Boolean -> float
    bool_features = [x for x in df.columns if df[x].dtype == 'bool']
    for feat in bool_features:
        df[feat] = df[feat].astype('float')

    # OneHot Encoding categorical data
    df = pd.get_dummies(df,
                        columns=['context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end'])

    split_df(df)
    return df


transformed_data = split_df(data)
# Wrangle the train and validation sets in the right format.
train_data, train_labels = transform(train)
val_data, val_labels = split_df(val)

train_data.shape, train_labels.shape

train_data.head()

# Drop the trackID columns from both the train/test data.

track_cols = [col for col in train_data.columns if 'track_id' in col[0]]
track_cols.append('track_id_clean')

train_data.drop(columns=track_cols, inplace=True)
val_data.drop(columns=track_cols, inplace=True)

train_data.head()

# Let's handle the missing values.
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

train_data_imp = imputer.fit_transform(train_data.drop(columns=['session_id']))
val_data_imp = imputer.transform(val_data.drop(columns=['session_id']))

# Shape of training data
print(f"Train Samples: {train_data_imp.shape}")
print(f'Labels: {train_labels.shape}')

"""##Model """


def evaluate(submission, groundtruth):
    """ Calculate metrics for prediction and ground thruth lists (source: starter kit) """
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0
    for sub, tru in zip(submission, groundtruth):
        if len(sub) != len(tru):
            raise Exception('Line {} should contain {} predictions, but instead contains '
                            '{}'.format(counter + 1, len(tru), len(sub)))
        try:
            ap_sum += ave_pre(sub, tru)
        except ValueError as e:
            raise ValueError('Invalid prediction in line {}, should be 0 or 1'.format(counter))
        first_pred_acc_sum += sub[0] == tru[0]
        counter += 1
    ap = ap_sum / counter
    first_pred_acc = first_pred_acc_sum / counter
    return ap, first_pred_acc


def ave_pre(submission, groundtruth):
    """ Calculate average accuracy (which is the same as average precision in this context) """
    s = 0.0
    t = 0.0
    c = 1.0
    for x, y in zip(submission, groundtruth):
        if x != 0 and x != 1:
            raise ValueError()
        if x == y:
            s += 1.0
            t += s / c
        c += 1
    return t / len(groundtruth)


def fit_predict(model):
    # Training the model
    model.fit(train_data_imp, train_labels)
    predictions = model.predict(val_data_imp)

    # Convert from flattened format back to session based format (for calculation of MAA metric).
    prediction_df = val_data[['session_id']]
    prediction_df.loc[:, 'prediction'] = predictions
    predictions_list = prediction_df.groupby('session_id')['prediction'].apply(list)

    labels_df = val_data[['session_id']]
    labels_df.loc[:, 'truth'] = val_labels
    truth_list = labels_df.groupby('session_id')['truth'].apply(list)

    ap, first_pred_acc = evaluate(predictions_list, truth_list)

    # Result
    print('average precision: {}'.format(ap))
    print('first prediction accuracy: {}\n\n'.format(first_pred_acc))


# Commented out IPython magic to ensure Python compatibility.
# %%time


# # Train GBT classifier
gbt_model = GradientBoostingClassifier(learning_rate=0.04,
                                       n_estimators=500)
# gbt_model.fit()
print(fit_predict(gbt_model))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from xgboost import XGBClassifier
#
# xgb_model = XGBClassifier(learning_rate= 0.05,
#                           n_estimators= 100)
# fit_predict(xgb_model)


# pickle.dump(gbt_model, open('model.pkl', 'wb'))