import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data_dependency import data_corelation_spring_layout


def sklearn_knn(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = KNeighborsClassifier()
    model.fit(x_train.astype(int),y_train.astype(int))
    print("modelScore of " + col_name + "is:  %.2f" % model.score(x_train.astype(int), y_train.astype(int)))
    y_pred = model.predict(x_test.astype(int))
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print("Accuracy of " + col_name + " is : %.2f%%" % (accuracy * 100.0))
    return accuracy



def randomforest_classifer(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train.astype(int), y_train.astype(int))
    print("modelScore of " + col_name + " is:  %.2f" % model.score(x_train.astype(int), y_train.astype(int)))
    y_pred = model.predict(x_test.astype(int))
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print("Accuracy of "+col_name+" is : %.2f%%" % (accuracy * 100.0))
    return accuracy





def knn_monte_carlo_using_data(file):
    df = pd.read_csv(file)
    accuracy_list = []

    column_name = list(df)
    model_list = []
    for col_name in column_name:
        x = df.loc[:, df.columns != col_name]
        # sc = StandardScaler()
        # x_scaler = sc.fit_transform(x)
        y = df[col_name]
        accuracy = sklearn_knn(x, y, col_name)
        accuracy_list.append(accuracy)
        #accuracy = randomforest_classifer(x, y, col_name)
        #accuracy_list.append(accuracy)
    s = pd.Series(
        accuracy_list,
        index=column_name
    )

    my_colors = 'rgbkymc'  # red, green, blue, black, etc.

    s.plot(
        kind='bar',

    )
    plt.show()


def knn_monte_carlo_using_data_depency_graph(file):
    G, df = data_corelation_spring_layout(file)
    accuracy_list = []
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
        accuracy = sklearn_knn(x, y, col_nm)
        accuracy_list.append(accuracy)
        # accuracy = randomforest_classifer(x, y, col_name)
        # accuracy_list.append(accuracy)
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
knn_monte_carlo_using_data('D:\Thesis\Sampled monte carlo Data from PF.csv')
#knn_monte_carlo_using_data('D:\Thesis\Sampled Realtime Data from PF.csv')

