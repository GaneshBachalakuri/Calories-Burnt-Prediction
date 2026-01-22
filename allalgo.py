import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('allalgo')
import pickle

class ALL_ALGO:
    def all_algorithems(X_train,X_test,y_train,y_test):
        try:
            reg_model=LinearRegression()
            reg_model.fit(X_train,y_train)
            logger.info(f'Intercept: {reg_model.intercept_}')
            logger.info(f'Coefficients: {reg_model.coef_}')
            y_train_pred_lr=reg_model.predict(X_train)
            a=pd.DataFrame({'ytrain':y_train,'ypred':y_train_pred_lr})
            logger.info(f'{a.sample(10)}')
            logger.info(X_train.columns)
            logger.info(f'Linear Regression r2_score: {r2_score(y_train, reg_model.predict(X_train))}')
            logger.info(f'Linear Regression loss: {mean_squared_error(y_train, reg_model.predict(X_train))}')
            logger.info(f'Linear Regression r2_score: {r2_score(y_test,reg_model.predict(X_test))}')
            logger.info(f'Linear Regression loss: {mean_squared_error(y_test,reg_model.predict(X_test))}')

            # logger.info(f'{reg_model.predict([[-0.163245,0.361072,0.236378,-0.826991,-1.595626 ,-0.667411,1.0]])}')

            with open('reg_model.pkl','wb') as f:
                pickle.dump(reg_model,f)

            with open('reg_model.pkl','rb') as f1:
                m = pickle.load(f1)
            with open('scalar.pkl','rb') as f2:
                s = pickle.load(f2)

            t = s.transform([[10,160,80,29,40,40]])
            print(t)
            t1 = np.append(t, [[500]], axis=1)

            logger.info(f" Prediction was: {m.predict(t1)}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')