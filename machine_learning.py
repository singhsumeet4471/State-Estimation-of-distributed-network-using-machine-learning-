import numpy
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from data_dependency import data_corelation_spring_layout


def linear_regression_using_data_depency_graph(file):
    G = data_corelation_spring_layout(file)
    nodes = list(G.nodes)
    node = []
    in_node =[]
    nodes_relation = pd.DataFrame()
    df_list = []

    for nd in nodes:

        for u, v in G.in_edges(nd, data=False):
           node.append(v)
           in_node.append(u)
        df = pd.DataFrame({'node':pd.Series(v), 'in_nodes':pd.Series(u)})
        df_list.append(df)

    final_df = pd.concat(df_list)
    print(final_df)

def linear_regression_using_data(file):
    df = pd.read_csv(file)
    column_name = list(df)
    model_list = []
    for col_name in column_name:
        x_train, x_test = train_test_split(df.loc[:, df.columns != col_name], test_size=0.2)
        y_train, y_test = train_test_split(df[col_name], test_size=0.2)
        print(x_train.shape,x_test.shape)
        print(y_train.shape,y_test.shape)
        # sc = StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # y_train = sc.transform(y_train)
        model  = MLPRegressor(hidden_layer_sizes=(25,),  activation='tanh', solver='adam')

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        #print('Coefficients for '+col_name +'\n', model.coef_)
        # The mean squared error
        print("Mean squared error for "+col_name+": %.2f"% mean_squared_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score for '+col_name+': %.2f' % r2_score(y_test, y_pred))


def baseline_model():
    model = Sequential()
    model.add(Dense(25, input_dim=39, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal'))
    model.add(Dense(units=1,kernel_initializer='normal',activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model



def linear_regression_using_keras(file):
        df = pd.read_csv(file)
        column_name = list(df)
        model_list = []
        #for col_name in column_name:
        # x = df.loc[:, 0:50].values
        # y = df.loc[:, 50].values
        x = df.loc[:, df.columns != 'p1']
        y = df['p1']
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






linear_regression_using_keras("D:\Thesis\Sensitivity analysis final.csv")