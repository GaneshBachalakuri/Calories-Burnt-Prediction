import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('missing_value_handling')
class MISSING_VAL:
    def random_sample(self):
        try:
            logger.info(f'shape training data:{X_train.shape}')
            logger.info(f'shape test data:{X_test.shape}')
            logger.info(f'before technique X_train:{X_train.columns}')
            logger.info(f'before technique X_test:{X_test.columns}')
            logger.info(f'before technique X_train:{X_train.isnull().sum()}')
            logger.info(f'before technique X_test:{X_test.isnull().sum()}')
            for i in X_train.columns:
                if X_train[i].isnull().sum() > 0:
                    logger.info(f'Train column name {i}')
                    X_train[i + "_replaced"] = X_train[i].copy()
                    X_test[i + "_replaced"] = X_test[i].copy()
                    s1 = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
                    s2 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                    s1.index = X_train[X_train[i].isnull()].index
                    s2.index = X_test[X_test[i].isnull()].index
                    X_train.loc[X_train[i].isnull(), i + "_replaced"] = s1
                    X_test.loc[X_test[i].isnull(), i + "_replaced"] = s2
                    X_train = X_train.drop([i], axis=1)
                    X_test = X_test.drop([i], axis=1)

            logger.info(f'after technique X_train:{X_train.columns}')
            logger.info(f'after technique X_test:{X_test.columns}')
            logger.info(f'after technique X_train:{X_train.isnull().sum()}')
            logger.info(f'after technique X_test:{X_test.isnull().sum()}')
            logger.info(f'total rows in training data:{X_train.shape}')
            logger.info(f'total rows in test data:{X_test.shape}')
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('missing_value_handling')
class MISSING_VAL:
    def random_sample(self):
        try:
            logger.info(f'shape training data:{X_train.shape}')
            logger.info(f'shape test data:{X_test.shape}')
            logger.info(f'before technique X_train:{X_train.columns}')
            logger.info(f'before technique X_test:{X_test.columns}')
            logger.info(f'before technique X_train:{X_train.isnull().sum()}')
            logger.info(f'before technique X_test:{X_test.isnull().sum()}')
            for i in X_train.columns:
                if X_train[i].isnull().sum() > 0:
                    logger.info(f'Train column name {i}')
                    X_train[i + "_replaced"] = X_train[i].copy()
                    X_test[i + "_replaced"] = X_test[i].copy()
                    s1 = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
                    s2 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                    s1.index = X_train[X_train[i].isnull()].index
                    s2.index = X_test[X_test[i].isnull()].index
                    X_train.loc[X_train[i].isnull(), i + "_replaced"] = s1
                    X_test.loc[X_test[i].isnull(), i + "_replaced"] = s2
                    X_train = X_train.drop([i], axis=1)
                    X_test = X_test.drop([i], axis=1)

            logger.info(f'after technique X_train:{X_train.columns}')
            logger.info(f'after technique X_test:{X_test.columns}')
            logger.info(f'after technique X_train:{X_train.isnull().sum()}')
            logger.info(f'after technique X_test:{X_test.isnull().sum()}')
            logger.info(f'total rows in training data:{X_train.shape}')
            logger.info(f'total rows in test data:{X_test.shape}')
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
