import numpy
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_dependency import data_corelation_spring_layout


def linear_regression(x,y,col_name):
    x_train, x_test = train_test_split(x, test_size=0.2,random_state=0)
    y_train, y_test = train_test_split(y, test_size=0.2,random_state=0)

    scalaer = StandardScaler().fit(x_train)
    x_train = scalaer.transform(x_train)
    x_test = scalaer.transform(x_test)

    model = LinearRegression().fit(x_train,y_train)
    #print(model.)
    print("modelScore of " + col_name + "is: %.2f" % model.score(x_train, y_train))
    y_pred = model.predict(x_test)
    # y_train.values.reshape(-1,1)
    # plt.scatter(x_train,y_train)
    # plt.plot(x_train,y_pred)
    # #x = linespace(10, 40, 5)
    # #plt.plot(x,x,'-')
    #plt.show()

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
    model = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam')
    model.fit(x_train,y_train)
    print("modelScore of "+col_name+"is: %.2f"% model.score(x_train,y_train))

    y_pred = model.predict(x_test)


def linear_regression_using_data_depency_graph(file):
    G,df = data_corelation_spring_layout(file)
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
        #linear_regression(x,y,col_nm)
        sklearn_MLPregressor(x, y, col_nm)

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


def linear_regression_using_data(file):
    df = pd.read_csv(file)
    column_name = list(df)
    model_list = []
    for col_name in column_name:
        x = df.loc[:, df.columns != col_name]
        # sc = StandardScaler()
        # x_scaler = sc.fit_transform(x)
        y = df[col_name]
        #linear_regression(x,y,col_name)
        sklearn_MLPregressor(x,y,col_name)







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








linear_regression_using_data_depency_graph("D:\Thesis\Sampled Realtime Data from PF.csv")