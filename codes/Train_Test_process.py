#############################################
# 10 - 03 - 2023
# @ Youssef ANNAKI
#############################################

from Feature_processing import feature_selection, zs_outliers_handling
from itertools import compress
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class train_test_process():

    def __init__(self, n_splits):
        """ Process variables. """

        self.n_splits = n_splits
        self.R2_train = {}
        self.R2_test = {}
        self.main_metric_train = {}
        self.main_metric_train_pvalue = {}
        self.main_metric_test = {}
        self.main_metric_test_pvalue = {}
        self.coeffs = {}
        self.predicted_train = []
        self.predicted_test = []


    def train_test(self, data, target, regression_model, model_params):
        """
            Rolling window train/test process.
             - model_params is dict of pipeline components parameters: {"model__arg1": param1, "model__arg2": param2, ...
                                                                        "feature_selection__arg1": param1,...}
        """

        period_length = len(data)
        interval_index = 1

        self.train_period = int((period_length / self.n_splits) * 0.7)
        self.test_period = int((period_length / self.n_splits) * 0.3)

        self.features_names = data.columns

        self.selected_features = {}  # Needed if a feature selection filter is applied.

        for i in range(0, period_length - (self.train_period + self.test_period),
                       10):  # (self.test_period + self.train_period) // 4

            """Logs. """

            print(f"Pointer on the rolling window number: {interval_index}")

            """ Split train/test sets. """

            self.X_train_set = data[i: i + self.train_period]
            self.X_test_set = data[i + self.train_period: i + self.train_period + self.test_period]

            self.Y_train_set = target[i: i + self.train_period]
            self.Y_test_set = target[i + self.train_period: i + self.train_period + self.test_period]

            """ Outliers elimination. """

            zs = zs_outliers_handling(self.X_train_set, self.Y_train_set, 3)  ## Hard coded threshold...
            self.X_train_set = zs["X_train"]
            self.Y_train_set = zs["Y_train"]

            """ Adjust Target variable. """

            self.Y_train_set = self.Y_train_set["TARGET"]
            self.Y_test_set = self.Y_test_set["TARGET"]

            """Logs. """

            print(
                f"Data set size: {period_length} \ntrain_set_size: {len(self.X_train_set)} \ntest_set_size: {len(self.X_test_set)} \n")


            """ Pipeline definition. """

            steps = [("scale", StandardScaler()),
                     ("feature_selection", feature_selection()),
                     ("model", regression_model),
                     ]
            self.pipe = Pipeline(steps)
            self.pipe.set_params(**model_params)

            """ Train model. """

            self.pipe.fit(self.X_train_set, self.Y_train_set)

            predicted_train = self.pipe.predict(self.X_train_set)
            predicted_test = self.pipe.predict(self.X_test_set)

            """ Metrics computation. """

            self.predicted_train += list(predicted_train)
            self.predicted_test += list(predicted_test)

            # Train r2
            self.R2_train[i + self.train_period] = r2_score(self.Y_train_set, predicted_train)

            # Correlation Train
            temp = stats.spearmanr(self.Y_train_set, predicted_train)

            self.main_metric_train[i + self.train_period] = temp.correlation
            self.main_metric_train_pvalue[i + self.train_period] = temp.pvalue

            # Test r2
            self.R2_test[i + self.test_period] = r2_score(self.Y_test_set, predicted_test)

            # Correlation Test
            temp = stats.spearmanr(self.Y_test_set, predicted_test)

            self.main_metric_test[i + self.test_period] = temp.correlation
            self.main_metric_test_pvalue[i + self.test_period] = temp.pvalue

            """ Selected features. """

            self.selected_features[i + self.train_period] = list(compress(self.features_names, self.pipe["feature_selection"].get_features()))



            interval_index += 1

            """Logs. """

            print("@" * 100 + "\n")

            # Coeffs stability

    #             for j in range(len(reg.coef_)):
    #                 if data.columns[j] not in self.coeffs:
    #                     self.coeffs[data.columns[j]] = {}

    #                 self.coeffs[data.columns[j]][i + self.train_period] = reg.coef_[j]

    def CV_train_test(self, data, target, regression_model, cv_param, scoring_type, task="get_model"):

        # Train model

        reg = regression_model  # .fit(data, target)

        # self.outliers_handling(data, target)
        #
        ppl = make_pipeline(SelectKBest(score_func=mutual_info_regression, k=31), preprocessing.StandardScaler(), reg)
        # ppl = make_pipeline(preprocessing.StandardScaler(), reg)

        # CV
        if task == "cv_score":
            scores = cross_val_score(ppl, data, target["TARGET"], cv=cv_param, scoring=scoring_type)

            self.cv_score_mean = scores.mean()
            self.cv_score_std = scores.std()

        else:
            self.cv_results = cross_validate(ppl, self.X_train_set, self.Y_train_set["TARGET"], cv=cv_param,
                                             scoring=('r2', 'neg_mean_squared_error'), return_estimator=True)

    def Hyperparam_tuning(self, model, data, target):
        """
            Hyperparameters tuning process.
            Given a data set and a model, perform the CVGridSearch defined for the given model.
            The user is free to set the different parameters' grids as well as the Cross-Validation scoring metric and to add more regression models following the same implementation structure.
             - model is a string in ["ElasticNet", "RF", "lgb", "XGB", "SVR", "ridge", "mlp", "lr", ]
        """

        def myscore(X, Y):
            temp = stats.spearmanr(X, Y)
            return temp.correlation

        self.score = make_scorer(myscore, greater_is_better = True)

        if model == "ElasticNet":

            parametersGrid = {"enet__max_iter": [10000, ],
                              "enet__alpha": [i / 100 for i in range(1, 500, 10)],
                              "enet__l1_ratio": [i / 100 for i in range(1, 100, 5)],
                              "selection__n_features": [i for i in range(1, len(data.columns), 5)],

                              }

            eNet = ElasticNet()
            ppl = Pipeline([("selection", feature_selection()), ("scaler", StandardScaler()), ("enet", eNet)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, error_score = 'raise',
                                     n_jobs = -1,
                                     )
            self.grid.fit(data, target["TARGET"])

        elif model == "RF":

            parametersGrid = {"rf__max_depth": [i for i in range(1, 20, 2)],
                              "rf__n_estimators": [i for i in range(1, 511, 100)],
                              "rf__min_samples_leaf": [i for i in range(1, 13, 2)],
                              "rf__max_features": [i / 10 for i in range(1, 10, 2)],
                              #"rf__bootstrap": [False, ],
                              "selection__n_features": [i for i in range(1, len(data.columns), 10)],
                              }

            RF = RandomForestRegressor()
            ppl = Pipeline([("selection", feature_selection()), ("rf", RF)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        elif model == "lgb":

            parametersGrid = {
                "lgb__task": ["predict", ],
                "lgb__application": ["regression", ],
                "lgb__objective": ["root_mean_squared_error", ],
                "lgb__boosting_type": ["gbdt", ],
                "lgb__learning_rate": [i / 10 for i in range(1, 13, 2)],
                #"lgb__num_leaves": [i for i in range(1, 20, 5)],
                #"lgb__tree_learner": ["serial", "feature", "data", "voting", ],
                "lgb__max_depth": [i for i in range(1, 20, 5)],
                #"lgb__min_data_in_leaf": [],
                "lgb__metric": ["rmse", ],
                #"lgb__feature_fraction": [],
                "lgb__n_estimators": [i for i in range(5, 1021, 50)],
                "lgb__reg_alpha": [i for i in range(0, 121, 20)],
                "lgb__reg_lambda": [i for i in range(0, 121, 20)],
                "lgb__n_jobs": [-1, ],
                "selection__n_features": [i for i in range(1, len(data.columns), 10)],
            }

            LGB = LGBMRegressor()
            ppl = Pipeline([("scaler", StandardScaler()), ("selection", feature_selection()), ("lgb", LGB)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, n_jobs = -1)
            self.grid.fit(data, target["TARGET"])




        elif model == "XGB":

            parametersGrid = {"xgb__max_depth": [i for i in range(1, 10, 2)],
                              "xgb__gamma": [i for i in range(1, 110, 10)],
                              "xgb__eta": [i / 10 for i in range(1, 13, 2)],
                              "xgb__reg_lambda": [i for i in range(1, 111, 10)],
                              "xgb__reg_alpha": [i for i in range(0, 111, 10)],
                              #"xgb__predictor": ["cpu_predictor", ],
                              "selection__n_features": [i for i in range(1, len(data.columns), 10)],
                              }

            XGB = XGBRegressor()
            ppl = Pipeline([("scaler", StandardScaler()), ("selection", feature_selection()), ("xgb", XGB)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        elif model == "SVR":

            parametersGrid = {"svr__kernel": ["rbf", ],
                              #"kernel": ["linear", "poly", "rbf", "sigmoid"],
                              "svr__C": [i / 10 for i in range(1, 1011, 10)],
                              "svr__epsilon": [i / 100 for i in range(1, 101, 5)],
                              "svr__shrinking": [True, ],
                              "svr__max_iter": [-1],
                              "selection__n_features": [i for i in range(1, len(data.columns) // 2, 5)],

                              }

            SV = SVR()
            ppl = Pipeline([("selection", feature_selection()), ("svr", SV)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, error_score = 'raise',
                                     n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        elif model == "ridge":

            parametersGrid = {"ridge__alpha": [i / 10 for i in range(-5000, 5000, 10)],
                              "selection__n_features": [i for i in range(1, len(data.columns))],

                              }

            ridge = Ridge()
            ppl = Pipeline([("selection", feature_selection()), ("scaler", StandardScaler()), ("ridge", ridge)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, error_score = 'raise',
                                     n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        elif model == "mlp":

            parametersGrid = {
                "mlp__hidden_layer_sizes": [(i, j) for i in range(10, 111, 10) for j in range(10, 111, 10)],
                "mlp__alpha": [i / 10 for i in range(1, 100, 10)],
                "mlp__max_iter": [10000, ],
                "selection__n_features": [i for i in range(1, len(data.columns), 5)],

                }

            mlp = MLPRegressor()

            ppl = Pipeline([("selection", feature_selection()), ("scaler", StandardScaler()), ("mlp", mlp)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 8, verbose = True, error_score = 'raise',
                                     n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        elif model == "lr":

            parametersGrid = {
                "selection__n_features": [i for i in range(1, len(data.columns))],

            }

            lr = LinearRegression()
            ppl = Pipeline([("selection", feature_selection()),
                            ("scaler", StandardScaler()), ("lr", lr)])

            self.grid = GridSearchCV(ppl, parametersGrid, scoring = self.score, cv = 5, verbose = True, error_score = 'raise',
                                     n_jobs = -1)
            self.grid.fit(data, target["TARGET"])


        """ logs. """

        print("GridSearchCV done. ")

    #def output_model(self, data, target, regression_model):

        #         self.feature_scaling(data, target)
    #    self.output_model_ = regression_model.fit(data, target["TARGET"])

    def plot(self):

        # R2 train
        plt.figure(figsize=(12, 7))
        plt.step(self.R2_train.keys(), list(self.R2_train.values()), color="b", linewidth=0.6)
        plt.show()

        # R2 test
        plt.figure(figsize=(12, 7))
        plt.step(self.R2_test.keys(), list(self.R2_test.values()), color="b", linewidth=0.6)
        plt.show()

        # Correlation train
        plt.figure(figsize=(12, 7))
        plt.step(self.main_metric_train.keys(), list(self.main_metric_train.values()), color="b", linewidth=0.6)
        plt.show()

        plt.figure(figsize=(12, 7))
        plt.step(self.main_metric_train_pvalue.keys(), list(self.main_metric_train_pvalue.values()), color="r",
                 linewidth=0.6)
        plt.show()

        # Correlation test
        plt.figure(figsize=(12, 7))
        plt.step(self.main_metric_test.keys(), list(self.main_metric_test.values()), color="b", linewidth=0.6)
        plt.show()

        plt.figure(figsize=(12, 7))
        plt.step(self.main_metric_test_pvalue.keys(), list(self.main_metric_test_pvalue.values()), color="r",
                 linewidth=0.6)
        plt.show()

        # Coeffs stability

#         plt.figure(figsize = (12, 7))
#         for feature in self.coeffs:
#             plt.step(self.coeffs[feature].keys(), list(self.coeffs[feature].values()), linewidth = 0.6, label = feature)

#         plt.legend()
#         plt.show()