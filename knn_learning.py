import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from data_dependency import data_corelation_spring_layout


def sklearn_knn(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto')
    model.fit(x_train.astype(int), y_train.astype(int))
    print("modelScore of " + col_name + "is:  %.2f" % model.score(x_train.astype(int), y_train.astype(int)))
    y_pred = model.predict(x_test.astype(int))
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print("Accuracy of " + col_name + " is : %.2f%%" % (accuracy * 100.0))
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    # measure = scalaer.inverse_transform(mean_abs_error)
    # print(measure)
    print("Mean absolute error of " + col_name + " is : %.2f%%" % mean_abs_error)
    precion = precision_score(y_test.astype(int), y_pred.astype(int), average='micro')
    recall = recall_score(y_test.astype(int), y_pred.astype(int), average='micro')
    f1score = f1_score(y_test.astype(int), y_pred.astype(int), average='micro')
    print("Mean precison , recall and F1 score of " + col_name + " is : %.2f%%" % precion, recall, f1score)
    return accuracy, mean_abs_error, precion, recall, f1score

def randomforest_classifer(x, y, col_name):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    # scalaer = StandardScaler().fit(x_train)
    # x_train = scalaer.transform(x_train)
    # x_test = scalaer.transform(x_test)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train.astype(int), y_train.astype(int))
    print("modelScore of " + col_name + " is:  %.2f" % model.score(x_train.astype(int), y_train.astype(int)))
    y_pred = model.predict(x_test.astype(int))
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print("Accuracy of " + col_name + " is : %.2f%%" % (accuracy * 100.0))
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    print("Mean absolute error of " + col_name + " is : %.2f%%" % mean_abs_error)
    precion = precision_score(y_test.astype(int), y_pred.astype(int), average='micro')
    recall = recall_score(y_test.astype(int), y_pred.astype(int), average='micro')
    f1score = f1_score(y_test.astype(int), y_pred.astype(int), average='micro')
    print("Mean precison , recall and F1 score of " + col_name + " is : %.2f%%" % precion, recall, f1score)
    return accuracy, mean_abs_error, precion, recall, f1score

def knn_monte_carlo_using_data(file):
    df = pd.read_csv(file)
    accuracy_list = []

    column_name = list(df)
    mean_abs_erro_list = []
    precison_list = []
    recall_list = []
    f1score_list = []
    for col_name in column_name:
        x = df.loc[:, df.columns != col_name]
        # sc = StandardScaler()
        # x_scaler = sc.fit_transform(x)
        y = df[col_name]
        accuracy,mean_error,precion, recall, f1score = sklearn_knn(x, y, col_name)
        # accuracy_list.append(accuracy)
        # mean_abs_erro_list.append(mean_error)
        #accuracy, mean_error, precion, recall, f1score = randomforest_classifer(x, y, col_name)
        accuracy_list.append(accuracy)
        mean_abs_erro_list.append(mean_error)
        precison_list.append(precion)
        recall_list.append(recall)
        f1score_list.append(f1score)
    s = pd.Series(
        accuracy_list,
        index=column_name
    )

    s.plot(
        kind='bar',

    )
    plt.show()
    df = pd.DataFrame({"model_name": pd.Series(column_name), "Accuracy Score": pd.Series(accuracy_list),
                       "Mean Absolute Error": pd.Series(mean_abs_erro_list), "precision": pd.Series(precison_list),
                       "Recall": pd.Series(recall_list),
                       "F1Score": pd.Series(f1score_list)})
    df.to_csv("D:\Thesis\score_sheet\Score_sheet_using_data.csv")

def knn_monte_carlo_using_data_depency_graph(file):
    G, df = data_corelation_spring_layout(file)
    accuracy_list = []
    mean_abs_erro_list = []
    precison_list = []
    recall_list = []
    f1score_list = []
    col_names = list(df['var1'].unique())
    data_df = pd.read_csv(file)
    # normalized_df = (data_df - data_df.mean()) / data_df.std()
    for col_nm in col_names:
        temp_df = pd.DataFrame(df.loc[df['var1'] == col_nm])
        xval = temp_df['var2']
        # yval = temp_df['node'].values
        x = data_df[xval]
        y = data_df[col_nm]
        # sc = StandardScaler()
        # x = sc.fit_transform(x_train)
        # accuracy,mean_error,precion, recall, f1score = sklearn_knn(x, y, col_nm)
        # accuracy_list.append(accuracy)
        # mean_abs_erro_list.append(mean_error)
        accuracy, mean_error, precion, recall, f1score = randomforest_classifer(x, y, col_nm)
        accuracy_list.append(accuracy)
        mean_abs_erro_list.append(mean_error)
        precison_list.append(precion)
        recall_list.append(recall)
        f1score_list.append(f1score)
    # width = 1
    # indexes = np.arange(len(col_names))
    # plt.bar(indexes, accuracy_list)
    # plt.xticks(indexes + width * 0.5, col_names)
    s = pd.Series(
        accuracy_list,
        index=col_names
    )

    s.plot(
        kind='bar',

    )

    plt.show()
    df = pd.DataFrame({"model_name": pd.Series(col_names), "Accuracy Score": pd.Series(accuracy_list),
                       "Mean Absolute Error": pd.Series(mean_abs_erro_list), "precision": pd.Series(precison_list),
                       "Recall": pd.Series(recall_list),
                       "F1Score": pd.Series(f1score_list)})
    df.to_csv("D:\Thesis\score_sheet\Score_sheet_using_data_dependency_graph.csv")

knn_monte_carlo_using_data('D:\Thesis\Training Data\Sampled monte carlo Data from PF.csv')
# knn_monte_carlo_using_data('D:\Thesis\Sampled Realtime Data from PF.csv')
