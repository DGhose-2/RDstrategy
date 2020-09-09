import pandas as pd
import numpy as np

# Feature preprocessing for predictive modelling

def feature_preprocess(df):
    # Process startdate for the Calendar Month of project start
    df['proj_month'] = pd.DatetimeIndex(df['startdate']).month

    # Process startdate for the year of project start, then as year after the first recorded in the data
    df['proj_year'] = pd.DatetimeIndex(df['startdate']).year
    first_year = min(df['proj_year'])
    df['proj_year'] = (df['proj_year']-first_year).astype('float')

    # Convert project_length to a pure number (of days)
    df['project_length'] = df['project_length'].dt.days.astype('float')

    # Let's just make sure CB_rank and projects_count are also numeric variables
    df[['CB_rank', 'projects_count']] = df[['CB_rank', 'projects_count']].astype('float')
    
    # We don't need org descriptions or proj titles any more as text
    df = df.drop(['description', 'title'], axis=1)
    
    return df

# For CNN:

def panel_maker(df_feat, df_lab, n_steps):
    n_features = len(df_feat.drop('startdate', axis=1).columns)

    X, y = [], []
    for i in range(len(df_feat)):
        # find panel end
        end_ix = i + n_steps
        # ensure not beyond the end of the dataset
        if end_ix > len(df_feat):
            break
        # gather input and output parts of the pattern
        seq_x = df_feat.iloc[i:end_ix] #index rows i:end_ix of feature dataset df1
        seq_y = df_lab.iloc[end_ix-1] #index row end_ix-1 of label dataset df2
        X.append(seq_x)
        y.append(seq_y)
    X_df = pd.concat(X).drop('startdate', axis=1).values.reshape((len(df_feat)-n_steps+1), n_steps, n_features)
    y_df = pd.DataFrame({'Occurred': y})
    return X_df, y_df


def final_preprocessing(df, model, n_steps=5):
    
    categorical_vars = ['orguuid', 'Occurred']
    df[categorical_vars] = df[categorical_vars].astype('category')
    
    if model=="dnn" or model=="ensemble":
        # drop extra features for time-independent analysis
        df = df.drop(['startdate', 'project_length', 'proj_month', 'proj_year'], axis=1)
    elif model=="cnn" or model=="lstm":
        # for time analysis (CNN), only include companies with projects_count >= 8
        df = df[df['projects_count']>=8]
        
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler
    
    # using labelencoder for orguuid category
    org_encod = LabelEncoder()
    df['orguuid'] = org_encod.fit_transform(df['orguuid'].values)
    n_orgs = df['orguuid'].nunique()
    
    if model=="cnn" or model=="lstm":
        #using ordinalencoder for proj_month category
        mon_encod = OrdinalEncoder()
        df['proj_month'] = mon_encod.fit_transform(df['proj_month'].values.reshape(-1,1))
        
    # Now split into training/validation and later-evaluation test sets
    df_trainval = df[df['typ']=='tv'].drop('typ', axis=1)
    df_test = df[df['typ']=='tst'].drop('typ', axis=1)


    # Normalization
    if model=="dnn" or model=="ensemble":
        numer_cols = ['CB_rank', 'projects_count', 'sim']
    elif model=="cnn" or model=="lstm":
        numer_cols = ['CB_rank', 'projects_count', 'sim', 'project_length', 'proj_year']
    numerics = df_trainval[numer_cols].values
    robustscaler = RobustScaler().fit(numerics)
    numerics_scaled = robustscaler.transform(numerics)
    df_trainval[numer_cols] = numerics_scaled
    
    # Normalize numeric variables according to TRAINING set mean and std
    numerics_test = df_test[numer_cols].values
    numerics_scaled = robustscaler.transform(numerics_test) #using the loaded mean and std from training-data
    df_test[numer_cols] = numerics_scaled

    if model=="dnn" or model=="ensemble":
        #Now split into feature data and labels
        X_trainval = df_trainval.drop(['Occurred'], axis=1)
        y_trainval = df_trainval['Occurred']

        #Split into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25)
    
    elif model=="cnn" or model=="lstm":
        # CNN and LSTM Preparation

        # First making the training-test split
        from sklearn.model_selection import train_test_split
        df_train, df_val = train_test_split(df_trainval, test_size=0.25)

        # Sort df_train and df_val by startdate within organization (without changing org position)
        import random
        def seq_maker(df):
            df = df.sort_values(['orguuid', 'startdate'], ascending = [True, True])
            grouper = [df for _, df in df.groupby('orguuid')]
            random.shuffle(grouper) #put the organizations in a random-order to avoid reading into their position, hence improve generality
            return pd.concat(grouper).reset_index(drop=True)

        df_train = seq_maker(df_train)
        df_val = seq_maker(df_val)

        # Sort df_test by startdate within organization (easier as org position does not leak future info this time)
        df_test = df_test.sort_values(['orguuid', 'startdate'], ascending = [True, True])
        
        if model=="cnn":
            # Now split into feature data and labels
            X_train = df_train.drop(['Occurred'], axis=1)
            y_train = df_train['Occurred']
            X_val = df_val.drop(['Occurred'], axis=1)
            y_val = df_val['Occurred']

            # Concatenate circumstantial windows for convolution
            X_train, y_train = panel_maker(X_train, y_train, n_steps=n_steps)

            # Same for val set
            X_val, y_val = panel_maker(X_val, y_val, n_steps=n_steps)
    
        elif model=="lstm":
            # now split into feature data and labels (start-date not needed for LSTM)
            X_train = df_train.drop(['Occurred', 'startdate'], axis=1)
            y_train = df_train['Occurred']
            X_val = df_val.drop(['Occurred', 'startdate'], axis=1)
            y_val = df_val['Occurred']


            # for siamese LSTM we must split inputs as follows:
            org_features = ['orguuid', 'CB_rank', 'projects_count']
            proj_features = ['project_length', 'sim', 'proj_month', 'proj_year']
            
            # reshape input to each LSTM
            X_train = [X_train[org_features].values.reshape(-1, 1, len(org_features)), X_train[proj_features].values.reshape(-1, 1, len(proj_features))]
            X_val = [X_val[org_features].values.reshape(-1, 1, len(org_features)), X_val[proj_features].values.reshape(-1, 1, len(proj_features))]
            
    return X_train, X_val, y_train, y_val, df_test, n_orgs


def run_dnn(X_train, X_val, y_train, y_val, df_test, n_orgs):

    # For use with organization embeddings later:
    X_train_array = [X_train['orguuid'], (X_train.drop('orguuid', axis=1))]
    X_val_array = [X_val['orguuid'], (X_val.drop('orguuid', axis=1))]

    import tensorflow as tf
    from tensorflow import keras

    # PReLU with constant
    from keras.initializers import Constant
    # embedding with L2 regularization
    from keras.regularizers import l2

    # # Gelu
    # from keras.utils.generic_utils import get_custom_objects
    # def gelu(x):
    #     return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    # get_custom_objects().update({'gelu': keras.layers.Activation(gelu)})
    
    n_latent_factors_orgs = 30    

    org_input = keras.layers.Input(shape=[1])
    org_embedding = keras.layers.Embedding(n_orgs + 1, n_latent_factors_orgs,
                                          embeddings_initializer='he_normal',
                                          embeddings_regularizer=l2(1e-6))(org_input)
    org_vec = keras.layers.Flatten()(org_embedding)
    org_vec = keras.layers.Dropout(0.2)(org_vec)

    other_input = keras.layers.Input(shape=(3,))

    concat = keras.layers.Concatenate()([org_vec, other_input])

    dense_1 = keras.layers.Dense(32, name='FullyConnected-1')(concat)
    act_1 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_1)
    dense_2 = keras.layers.Dense(16, name='FullyConnected-2')(act_1)
    act_2 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_2)
    dense_3 = keras.layers.Dense(16, name='FullyConnected-3')(act_2)
    act_3 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_3)
    dense_4 = keras.layers.Dense(16, name='FullyConnected-4')(act_3)
    act_4 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_4)
    output = keras.layers.Dense(1, activation=tf.nn.sigmoid, name='Output')(act_4)

    model = keras.Model([org_input, other_input], output)

    model.compile(optimizer='adam',              
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_array, y_train, epochs=20, batch_size=1, verbose=1, validation_data=(X_val_array, y_val))
    
    val_loss, val_acc = model.evaluate(X_val_array, y_val)
    print('Validation accuracy:', val_acc)
    
    rank_results = test_results(df_test, alg="dnn", model=model)
    return rank_results


def run_ensemble(X_train, X_val, y_train, y_val, df_test):

    ### ENSEMBLE LEARNING with (naive) classification models

    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    import xgboost as xgb

    final_layer = StackingClassifier(
        estimators=[('knn', KNeighborsClassifier(n_neighbors=6))],
        final_estimator=xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    )
    model = StackingClassifier(
        estimators=[('rf', RandomForestClassifier(random_state=42)),
                    ('svc', SVC(C=1, gamma=1e-6, kernel='rbf')),
                   ],
        final_estimator=final_layer
    )

    history = model.fit(X_train, y_train)

    print(accuracy_score(y_val, model.predict(X_val)))
    
    rank_results = test_results(df_test, alg="ensemble", model=model)
    return rank_results



def run_cnn(X_train, X_val, y_train, y_val, df_test, n_steps):
    import tensorflow as tf
    from tensorflow import keras

    # PReLU with constant
    from keras.initializers import Constant
    # Embedding with L2 regularization
    from keras.regularizers import l2

    n_features = len(df_test.drop(['Occurred','startdate'], axis=1).columns) #input features only

    cnn_input = keras.layers.Input(shape=(n_steps, n_features))
    conv_1 = keras.layers.Conv1D(filters=64, kernel_size = n_steps, activation='relu')(cnn_input)
    max_pool_1 = keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv_1)

    dense_1 = keras.layers.Dense(32, name='FullyConnected-1')(max_pool_1)
    act_1 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_1)
    dense_2 = keras.layers.Dense(16, name='FullyConnected-2')(act_1)
    act_2 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_2)
    dense_3 = keras.layers.Dense(16, name='FullyConnected-3')(act_2)
    act_3 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(dense_3)
    output = keras.layers.Dense(1, activation=tf.nn.sigmoid, name='Output')(act_3)

    model = keras.Model(cnn_input, output)

    model.compile(optimizer='adam',              
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1, validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print('Validation accuracy:', val_acc)
    
    rank_results = test_results(df_test, alg="cnn", model=model, n_steps=n_steps)
    return rank_results


# LSTM model

# Manhattan distance similarity function in output layer
from keras import backend as K
from keras.layers import Layer

class ManDist(Layer):    # adapted from https://github.com/sdcslyz/SummerProj_ZJU/blob/master/DL_model.py
    def __init__(self, **kwargs):
            self.res = None
            super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.res  = K.exp(- K.sum(K.abs(inputs[0]-inputs[1]), axis = 1, keepdims = True))
        return self.res

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.res)

# Training the siamese BiLSTMs

def run_lstm(X_train, X_val, y_train, y_val, df_test, org_features, proj_features):
    import tensorflow as tf
    from tensorflow import keras

    # PReLU with constant
    from keras.initializers import Constant
    #Embedding with L2 regularization
    from keras.regularizers import l2
        
    # LSTM_a
    org_input = keras.layers.Input(shape=(None, len(org_features)))
    lstm_org_1 = keras.layers.Bidirectional(keras.layers.LSTM(32, name='Left-LSTM'))(org_input)
    org_vec = keras.layers.Dropout(0.2, name='Left-Dropout')(lstm_org_1)
    org_dense_1 = keras.layers.Dense(32, name='Left-FullyConnected-1')(org_vec)
    org_act_1 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(org_dense_1)
    org_dense_2 = keras.layers.Dense(16, name='Left-FullyConnected-2')(org_act_1)
    org_act_2 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(org_dense_2)
    org_fin = keras.layers.Flatten()(org_act_2)

    # LSTM_b
    proj_input = keras.layers.Input(shape=(None, len(proj_features)))
    lstm_proj_1 = keras.layers.Bidirectional(keras.layers.LSTM(32, name='Right-LSTM'))(proj_input)
    proj_vec = keras.layers.Dropout(0.2, name='Right-Dropout')(lstm_proj_1) 
    proj_dense_1 = keras.layers.Dense(32, name='Right-FullyConnected-1')(proj_vec)
    proj_act_1 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(proj_dense_1)
    proj_dense_2 = keras.layers.Dense(16, name='Right-FullyConnected-2')(proj_act_1)
    proj_act_2 = keras.layers.PReLU(alpha_initializer=Constant(value=0.25))(proj_dense_2)
    proj_fin = keras.layers.Flatten()(proj_act_2)

    output = ManDist(name='Combined-Similarity')([org_vec, proj_vec])

    model = keras.Model([org_input, proj_input], output)

    model.compile(optimizer='adam',              
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=12, batch_size=1, verbose=1, validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print('Validation accuracy:', val_acc)
    
    rank_results = test_results(df_test, alg="lstm", model=model, org_features=org_features, proj_features=proj_features)
    return rank_results


def predict(df_test, alg, model, n_steps=5, org_features=['orguuid', 'CB_rank', 'projects_count'], proj_features=['project_length', 'sim', 'proj_month', 'proj_year']):
    X_test = df_test.drop(['Occurred'], axis=1)
    y_test = df_test['Occurred']
    
    if alg=="dnn":
        X_test_array = [X_test['orguuid'], (X_test.drop('orguuid', axis=1))]

        # generate predictions on the test inputs
        y_pred = model.predict(X_test_array)
        
    elif alg=="ensemble":
        # generate predictions on the test inputs
        y_pred = model.predict_proba(X_test)[:,1] #we want second column for probability of a 1
    
    elif alg=='rand_bench':
        # adding column of random numbers
        y_pred = np.random.uniform(low=0.0, high=1.0, size=len(df_test)).tolist()
    
    elif alg=="cnn":
        X_test_cnn, y_test_cnn = panel_maker(X_test, y_test, n_steps)
        
        # generate predictions on the test inputs
        y_pred = model.predict(X_test_cnn)
        y_pred = np.ndarray.flatten(y_pred)
    
    elif alg=="lstm":
        X_test = X_test.drop(['startdate'], axis=1)
        X_test_array = [X_test[org_features].values.reshape(-1, 1, len(org_features)), X_test[proj_features].values.reshape(-1, 1, len(proj_features))]
        
        # generate predictions on the test inputs
        y_pred = model.predict(X_test_array)
        
    return y_pred 



def test_results(df_test, alg, model, n_steps=5, org_features=['orguuid', 'CB_rank', 'projects_count'], proj_features=['project_length', 'sim', 'proj_month', 'proj_year']):
    # generate predictions from the model
    y_pred = predict(df_test=df_test, alg=alg, model=model, n_steps=n_steps, org_features=org_features, proj_features=proj_features)
    
    # for CNN only, remove top fringe/edge, for which we don't make sequential predictions
    if alg=="cnn":
        df_test = df_test.iloc[(n_steps-1):]

    # append the predictions to the test-set
    df_test = df_test.assign(Occurred_Pred = y_pred)
    
    # Get lists of true Occurred, in descending order of how likely the model thinks they are to have occurred
    if alg=="dnn" or alg=="rand_bench" or alg=="cnn" or alg=="lstm":
        pred_ranks = df_test.sort_values(['orguuid', 'Occurred_Pred'], ascending=[True, False]).groupby('orguuid')['Occurred'].apply(list).reset_index(drop=False)
    elif alg=="ensemble":
        rand_nums = np.random.uniform(low=0.0, high=1.0, size=len(df_test)).tolist()
        df_test = df_test.assign(extra = rand_nums)
        pred_ranks = df_test.sort_values(['orguuid', 'Occurred_Pred', 'extra'], ascending=[True, False, False]).groupby('orguuid')['Occurred'].apply(list).reset_index(drop=False)

    # Get a dataframe with the ranking positions of true examples against total count per company
    pred_ranks['occurred_pos'] = pred_ranks['Occurred'].apply(lambda x: np.where(x)[0].tolist())
    pred_ranks['occurred_count'] = pred_ranks['occurred_pos'].str.len()

    # MAP@k is available in ml_metrics but MAR@k, which is more important for our purposes, is not. I write my own functions for both

    # AP@k
    pred_ranks['Avg_Prec_at_k'] = pred_ranks['Occurred'].apply(lambda x: sum((np.cumsum(x)/range(1,len(x)+1)*x).tolist())).div(pred_ranks['occurred_count'].values,axis=0)

    # AR@k
    pred_ranks['Avg_Rec_at_k'] = pred_ranks['Occurred'].apply(lambda x: sum(np.divide((np.cumsum(x)*x), x.count(1)).tolist())).div(pred_ranks['occurred_count'].values,axis=0)

    # DCG@k
    pred_ranks['DCG'] = pred_ranks['Occurred'].apply(lambda x: sum((np.divide(np.array([(2**y-1) for y in x], dtype=np.float),
                                                                              np.array([(np.log2(y+1)) for y in range(1,len(x)+1)], dtype=np.float))).tolist()))
   
    # collect means for the metrics above
    rank_results = pred_ranks.groupby('occurred_count')[['Avg_Prec_at_k', 'DCG']].mean().reset_index(drop=False)

    # use a more consistent evaluation metric than DCG, which is DCG divided by occurred_count
    rank_results['fDCG'] = rank_results['DCG']/rank_results['occurred_count']
    rank_results = rank_results.drop(['DCG'], axis=1)
    
    return rank_results
