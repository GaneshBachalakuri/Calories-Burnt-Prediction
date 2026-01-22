import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('cat_to_num')

class CATEGORYTONUM:
    def category_to_numeric(X_train,X_test):
        try:
            logger.info(f'{X_train.columns}')
            for i in X_train.columns:
                logger.info(f'{i}--->{X_train[i].unique()}')
            one_hot=OneHotEncoder(drop='first')
            one_hot.fit(X_train[['Gender']])
            result=one_hot.transform(X_train[['Gender']]).toarray()
            f=pd.DataFrame(data=result,columns=one_hot.get_feature_names_out())
            X_train.reset_index(drop=True,inplace=True)
            f.reset_index(drop=True,inplace=True)
            X_train=pd.concat([X_train,f],axis=1)
            #logger.info(f'Before{X_train.columns}')
            X_train=X_train.drop(['Gender'],axis=1)
            logger.info(f'{X_train.columns}')
            #logger.info(f'{X_train}')

            result1=one_hot.transform(X_test[['Gender']]).toarray()
            f1=pd.DataFrame(data=result1,columns=one_hot.get_feature_names_out())
            X_test.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            X_test=pd.concat([X_test,f1],axis=1)
            #logger.info(f'Before{X_test.columns}')
            X_test=X_test.drop(['Gender'],axis=1)
            logger.info(f'{X_test.columns}')
            #logger.info(f'{X_test}')
            return X_train,X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')