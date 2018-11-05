from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_dependency import data_absolute_diff_network_grid_layout


def baseline_model(x,y):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = Sequential()
    model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal'))
    model.add(Dense(units=1,kernel_initializer='normal',activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=128,epochs=40)
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
        x = df.loc[:, df.columns != col_name]

        y = df[col_name]
        model = baseline_model(x,y)



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




keras_nn_using_data_depency_graph("D:\Thesis\Sensitivity analysis final.csv")