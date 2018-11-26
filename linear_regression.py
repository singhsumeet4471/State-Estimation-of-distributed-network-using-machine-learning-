import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_dependency import data_corelation_spring_layout


def bar_graph(label_list,value_list):
    s = pd.Series(
        value_list,
        index=label_list
    )

    s.plot(
        kind='bar',

    )

    plt.show()


def linear_regression(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2,random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2,random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = LinearRegression(normalize=False)
    model.fit(x_train,y_train)
    #print(model.)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test.astype(int),y_pred.astype(int))
    print("Accuracy of " + col_name + " is : %.2f%%" % (accuracy * 100.0))
    mean_abs_error = mean_absolute_error(y_test,y_pred)
    print("Mean absolute error of " + col_name + " is : %.2f%%" % mean_abs_error)
    precion= precision_score(y_test.astype(int),y_pred.astype(int),average='micro')
    recall = recall_score(y_test.astype(int),y_pred.astype(int),average='micro')
    f1score = f1_score(y_test.astype(int),y_pred.astype(int),average='micro')
    print("Mean precison , recall and F1 score of " + col_name + " is : %.2f%%" % precion,recall,f1score)
    return accuracy,mean_abs_error,precion,recall,f1score


def baseline_model():
    model = Sequential()
    model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal'))
    model.add(Dense(units=1,kernel_initializer='normal',activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model


def sklearn_MLPregressor(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)
    model = MLPRegressor(hidden_layer_sizes=(25,), activation='relu', solver='adam')
    model.fit(x_train,y_train)
    print("modelScore of "+col_name+" is: %.2f"% model.score(x_train,y_train))


    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print("Accuracy of " + col_name + " is : %.2f%%" % (accuracy * 100.0))
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    print("Mean absolute error of " + col_name + " is : %.2f%%" % mean_abs_error)
    precion = precision_score(y_test.astype(int), y_pred.astype(int),average='micro')
    recall = recall_score(y_test.astype(int), y_pred.astype(int),average='micro')
    f1score = f1_score(y_test.astype(int), y_pred.astype(int),average='micro')
    print("Mean precison , recall and F1 score of " + col_name + " is : %.2f%%" % precion, recall, f1score)
    return accuracy,mean_abs_error,precion, recall, f1score


def linear_regression_using_data_depency_graph(file):
    G,df = data_corelation_spring_layout(file)
    accuracy_list =[]
    mean_abs_erro_list = []
    precison_list =[]
    recall_list =[]
    f1score_list =[]
    col_names = list(df['var1'].unique())
    data_df = pd.read_csv(file)
    normalized_df = (data_df - data_df.mean()) / data_df.std()
    for col_nm in col_names:
        temp_df = pd.DataFrame(df.loc[df['var1'] == col_nm])
        xval = temp_df['var2']
        #yval = temp_df['node'].values
        x = normalized_df[xval]
        y = normalized_df[col_nm]
        #sc = StandardScaler()
        #x = sc.fit_transform(x_train)
        acuuracy, mean_error, precion, recall, f1score = linear_regression(x,y,col_nm)
        # accuracy_list.append(acuuracy)
        # mean_abs_erro_list.append(mean_error)
        #acuuracy, mean_error,precion, recall, f1score = sklearn_MLPregressor(x, y, col_nm)
        accuracy_list.append(acuuracy)
        mean_abs_erro_list.append(mean_error)
        precison_list.append(precion)
        recall_list.append(recall)
        f1score_list.append(f1score)

        # seed = 42
        # numpy.random.seed(seed)
        # # evaluate model with standardized dataset
        # estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=0)
        # kfold = KFold(n_splits=10, random_state=seed)
        # results = cross_val_score(estimator, x, y, cv=kfold)
        # print("Results: %.2f (%.2f) MSE " % (results.mean(), results.std()))
        # # serialize model to JSON
        # model_json = estimator.to_json()
        # with open("D:\Thesis\model\model"+col_nm+".json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # estimator.save_weights("model.h5")
        # print("Saved model to disk")
    bar_graph(col_names, accuracy_list)
    df = pd.DataFrame({"model_name": pd.Series(col_names), "Accuracy Score": pd.Series(accuracy_list),
                       "Mean Absolute Error": pd.Series(mean_abs_erro_list),"precision":pd.Series(precison_list),"Recall":pd.Series(recall_list),
                       "F1Score":pd.Series(f1score_list)})
    df.to_csv("D:\Thesis\score_sheet\Score_sheet_using_data_depencency graph.csv")


def linear_regression_using_data(file):
    df = pd.read_csv(file)
    column_name = list(df)
    mean_abs_error_list = []
    accuracy_list = []
    precison_list = []
    recall_list = []
    f1score_list = []
    for col_name in column_name:
        x = df.loc[:, df.columns != col_name]
        # sc = StandardScaler()
        # x_scaler = sc.fit_transform(x)
        y = df[col_name]
        #acuuracy, mean_error, precion, recall, f1score = linear_regression(x, y, col_name)
        acuuracy, mean_error,precion, recall, f1score  =sklearn_MLPregressor(x,y,col_name)
        accuracy_list.append(acuuracy)
        mean_abs_error_list.append(mean_error)
        precison_list.append(precion)
        recall_list.append(recall)
        f1score_list.append(f1score)

    bar_graph(column_name,accuracy_list)

    df = pd.DataFrame({"model_name":pd.Series(column_name),"Accuracy Score":pd.Series(accuracy_list),
                       "Mean Absolute Error":pd.Series(mean_abs_error_list),"precision":pd.Series(precison_list),"Recall":pd.Series(recall_list),
                       "F1Score":pd.Series(f1score_list)})
    df.to_csv("D:\Thesis\score_sheet\Score_sheet_using_data.csv")



def linear_regression_using_keras(file):
    df = pd.read_csv(file)
    column_name = list(df)
    model_list = []
    for col_name in column_name:
        # x = df.loc[:, 0:50].values
        # y = df.loc[:, 50].values
        x = df.loc[:, df.columns != col_name]
        y = df[col_name]
        print(x.shape)
        print(y.shape)


        #x_train, x_test = train_test_split(df.loc[:, df.columns != 'p1'], test_size=0.2)
        #y_train, y_test = train_test_split(df['p1'], test_size=0.2)
        seed = 42
        numpy.random.seed(seed)
        # evaluate model with standardized dataset
        estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=0)
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(estimator, x, y, cv=kfold)
        print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))








linear_regression_using_data_depency_graph("D:\Thesis\Training Data\Sampled Realtime Data from PF.csv")