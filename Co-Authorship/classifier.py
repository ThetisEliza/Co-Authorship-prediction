from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from tensorflow import keras



class Classifier:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def get_result(self):
        df = self.data_frame
        train_x = df.tr_x
        train_y = df.tr_y
        test_x = df.te_x
        test_y = df.te_y
        pred_x = df.pr_x


        # clf = LogisticRegression(C=1, max_iter=1000, )
        # # clf = DecisionTreeRegressor()
        # clf.fit(train_x, train_y)
        # predictions = clf.predict_proba(test_x)[:, 1]
        # print('AUC', roc_auc_score(test_y, predictions))
        #
        # pred_y = clf.predict_proba(pred_x)[:, 1]
        #
        # pred = clf.predict(pred_x)
        #
        # print(pred)


        self.model = keras.models.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(train_x.shape[1],)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(4, activation='relu'),

            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            loss='mse',
            metrics=['mse', 'accuracy'])

        model = self.model
        model.fit(train_x, train_y, epochs=10)

        model = self.model
        cost = model.evaluate(test_x, test_y)
        print('eva', cost)
        Y_pred = model.predict(test_x)
        print(Y_pred[:, 0])
        print(roc_auc_score(test_y, Y_pred[:, 0]))
        model = self.model
        DATA_X = pred_x
        # probability_model = keras.Sequential([model,
        #                                       keras.layers.Softmax()])
        Y_pred = model.predict(DATA_X)
        return Y_pred[:, 0]




        return pred_y