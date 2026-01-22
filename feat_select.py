import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('feat_select')
from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold

class FEATURE:
    def feature_selecting(X_train,X_test,y_train):
        try:
            logger.info(f'{X_train.columns}-->{X_train.shape}')
            logger.info(f'{X_test.columns}-->{X_test.shape}')

            # constant
            reg_con = VarianceThreshold(threshold=0.0)
            reg_con.fit(X_train)
            logger.info(f'columns we need to remove from constant technique :{X_train.columns[~reg_con.get_support()]}')
            good_data = reg_con.transform(X_train)
            good_data_1 = reg_con.transform(X_test)
            X_train_fs = pd.DataFrame(data=good_data,columns=X_train.columns[reg_con.get_support()])
            X_test_fs = pd.DataFrame(data=good_data_1,columns=X_test.columns[reg_con.get_support()])

            # quasi constant
            reg_quasi = VarianceThreshold(threshold=0.1)
            reg_quasi.fit(X_train_fs)
            logger.info(f'columns we need to remove from quasi constant technique :{X_train_fs.columns[~reg_quasi.get_support()]}')
            good_data_2 = reg_quasi.transform(X_train_fs)
            good_data_3 = reg_quasi.transform(X_test_fs)
            X_train_fs_1 = pd.DataFrame(data=good_data_2, columns=X_train_fs.columns[reg_quasi.get_support()])
            X_test_fs_2 = pd.DataFrame(data=good_data_3, columns=X_test_fs.columns[reg_quasi.get_support()])
            logger.info(f'{X_train_fs_1.columns}-->{X_train_fs_1.shape}')
            logger.info(f'{X_test_fs_2.columns}-->{X_test_fs_2.shape}')

            # Hypothesis testing
            logger.info(f'{y_train.unique()}')
            values=[]
            for i in X_train_fs_1.columns:
                values.append(pearsonr(X_train_fs_1[i], y_train))
            values=np.array(values)
            p_values=pd.Series(values[:, 1],index=X_train_fs_1.columns)
            p_values.sort_values(ascending=True,inplace=True)

            #logger.info(f'{p_values}')
            #p_values.plot.bar()
            #plt.show()

            alpha=0.05
            drop_cols=p_values[p_values>alpha].index
            logger.info(f'{drop_cols}')
            logger.info(f'=================================================')
            logger.info(f'Before :{X_train_fs_1.columns}-->{X_train_fs_1.shape}')
            logger.info(f'Before :{X_test_fs_2.columns}-->{X_test_fs_2.shape}')
            X_train_fs_1 = X_train_fs_1.drop(columns=drop_cols, axis=1)
            X_test_fs_2 = X_test_fs_2.drop(columns=drop_cols, axis=1)
            logger.info(f'After :{X_train_fs_1.columns}-->{X_train_fs_1.shape}')
            logger.info(f'After :{X_test_fs_2.columns}-->{X_test_fs_2.shape}')

            return X_train_fs_1, X_test_fs_2

        except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')