import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('data_scaling')
import pickle
from sklearn.preprocessing import StandardScaler

class DATASCALE:
    def data_scale(X_train,X_test):
        try:
            logger.info(f'\n{X_train}')
            logger.info(f'\n{X_test}')
            logger.info(f'\n{X_train.columns}')
            sc=StandardScaler()
            sc.fit(X_train)
            X_train_scal=sc.transform(X_train)
            X_test_scal=sc.transform(X_test)
            logger.info(f"\n{X_train_scal}")
            logger.info(f"\n{X_test_scal}")
            X_train_scal=pd.DataFrame(data=X_train_scal,columns=X_train.columns)
            X_test_scal=pd.DataFrame(data=X_test_scal,columns=X_test.columns)
            logger.info(f"\n{X_train_scal}")
            logger.info(f"\n{X_test_scal}")
            with open('scalar.pkl','wb') as f:
                pickle.dump(sc,f)
            return X_train_scal,X_test_scal
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')