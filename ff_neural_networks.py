
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_dependency import data_absolute_diff_network_grid_layout


def baseline_model(x,y):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    # scalaer = StandardScaler().fit(x_train)
    # x_train = scalaer.transform(x_train)
    # x_test = scalaer.transform(x_test)

    model = Sequential()
    model.add(Dense(20, input_dim=49, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.2)
    print(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return model


# def neural_networks():
#     seed = 42
#     np.random.seed(seed)
#     # evaluate model with standardized dataset
#
#     kfold = KFold(n_splits=10, random_state=seed)
#     results = cross_val_score(estimator, x, y, cv=kfold)
#     print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#     accuracy_list.append(accuracy)
#     mean_abs_erro_list.append(mean_error)


def keras_nn_using_data(file):
    df = pd.read_csv(file)
    column_name = list(df)
    mean_abs_error_list = []
    accuracy_list = []
    for col_name in column_name:
        x = df.loc[:, df.columns != col_name].values
        y = df[col_name].values
        y = np.reshape(y, (-1, 1))
        scaler = MinMaxScaler()
        print(scaler.fit(x))
        print(scaler.fit(y))
        xscale = scaler.transform(x)
        yscale = scaler.transform(y)
        baseline_model(xscale,yscale)



    # bar_graph(column_name, accuracy_list)
    #
    # df = pd.DataFrame({"model_name": pd.Series(column_name), "Accuracy Score": pd.Series(accuracy_list),
    #                    "Mean Absolute Error": pd.Series(mean_abs_error_list)})
    # df.to_csv("D:\Thesis\score_sheet\Score_sheet_using_data.csv")

def keras_nn_using_data_depency_graph(file):
    G, df = data_absolute_diff_network_grid_layout(file)
    accuracy_list = []
    mean_abs_erro_list = []
    col_names = list(df['var1'].unique())
    data_df = pd.read_csv(file)
    normalized_df = (data_df - data_df.mean()) / data_df.std()
    for col_nm in col_names:
        temp_df = pd.DataFrame(df.loc[df['var1'] == col_nm])
        xval = temp_df['var2']
        # yval = temp_df['node'].values
        x = normalized_df[xval]
        y = normalized_df[col_nm]
        # sc = StandardScaler()
        # x = sc.fit_transform(x_train)
        # accuracy,mean_error = sklearn_knn(x, y, col_nm)
        # accuracy_list.append(accuracy)
        # mean_abs_erro_list.append(mean_error)
        model = baseline_model(x, y)




keras_nn_using_data("D:\Thesis\Backup_Dataset\Sampled monte carlo Data from PF.csv")