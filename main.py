import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from missing_value_handling import MISSING_VAL
from variable_trans_out_handle import VAR_TRANS_OUT_HANDLE
from cat_to_num import CATEGORYTONUM
from feat_select import FEATURE
from data_scaling import DATASCALE
from allalgo import ALL_ALGO


class CALORIES_BURNT_PREDICTION:
    def __init__(self,path1,path2):
        try:
            self.path1=path1
            self.path2=path2
            self.df1=pd.read_csv(self.path1)
            self.df2=pd.read_csv(self.path2)
            logger.info(f'shape of df1:{self.df1.shape}')
            logger.info(f'shape of df2:{self.df2.shape}')
            logger.info(f'common columns :{self.df1.columns.intersection(self.df2.columns)}')
            self.df=pd.merge(self.df2,self.df1,on='User_ID',how='right')
            logger.info(f'{self.df.columns}')
            logger.info(f'shape of df:{self.df.shape}')
            logger.info(f'{self.df.isnull().sum()}')
            for i in self.df.columns:
                logger.info(f'{i}-->{self.df[i].dtype}')
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0:
                    logger.info(f'{i}-->{self.df[i].dtype}')
                    if self.df[i].dtype == object:
                        self.df[i] = pd.to_numeric(self.df[i])
                        logger.info(f'{i}-->{self.df[i].dtype}')
                    else:
                        pass
            self.X=self.df.iloc[:,1:-1]
            self.y=self.df.iloc[:,-1]
            logger.info(f'shape of X:{self.X.shape}')
            logger.info(f'shape of y:{self.y.shape}')
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'shape of X_train:{self.X_train.shape}')
            logger.info(f'shape of X_test:{self.X_test.shape}')
            logger.info(f'shape of y_train:{self.y_train.shape}')
            logger.info(f'shape of y_test:{self.y_test.shape}')
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            logger.info(f'columns of y_train:{self.y_train.name}')
            logger.info(f'columns of y_test:{self.y_test.name}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def missing_values(self):
        try:
            if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                self.X_train,self.X_test=MISSING_VAL.random_sample(self.X_train,self.X_test)
            else:
                logger.info(f'There are no missing values in X_train and X_test')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def vt_outhand(self):
        try:
            #logger.info(f'{self.X_train.info()}')
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            self.X_train_num=self.X_train.select_dtypes(exclude=object)
            self.X_train_cat=self.X_train.select_dtypes(include=object)
            self.X_test_num=self.X_test.select_dtypes(exclude=object)
            self.X_test_cat=self.X_test.select_dtypes(include=object)
            logger.info(f'columns of X_train_num:{self.X_train_num.columns}')
            logger.info(f'columns of X_train_cat:{self.X_train_cat.columns}')
            logger.info(f'columns of X_test_num:{self.X_test_num.columns}')
            logger.info(f'columns of X_test_cat:{self.X_test_cat.columns}')
            logger.info(f'shape of X_train_num:{self.X_train_num.shape}')
            logger.info(f'shape of X_train_cat:{self.X_train_cat.shape}')
            logger.info(f'shape of X_test_num:{self.X_test_num.shape}')
            logger.info(f'shape of X_test_cat:{self.X_test_cat.shape}')
            self.X_train_num,self.X_test_num=VAR_TRANS_OUT_HANDLE.variable_transform_outliers(self.X_train_num,self.X_test_num)
            logger.info(f'===========================================================')
            logger.info(f'columns of X_train_num:{self.X_train_num.columns}')
            logger.info(f'columns of X_train_cat:{self.X_train_cat.columns}')
            logger.info(f'columns of X_test_num:{self.X_test_num.columns}')
            logger.info(f'columns of X_test_cat:{self.X_test_cat.columns}')
            logger.info(f'shape of X_train_num:{self.X_train_num.shape}')
            logger.info(f'shape of X_train_cat:{self.X_train_cat.shape}')
            logger.info(f'shape of X_test_num:{self.X_test_num.shape}')
            logger.info(f'shape of X_test_cat:{self.X_test_cat.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def categori_num(self):
        try:
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            self.X_train_cat,self.X_test_cat=CATEGORYTONUM.category_to_numeric(self.X_train_cat,self.X_test_cat)
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f"{self.X_train_cat.shape}")
            logger.info(f"{self.X_test_cat.shape}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def feature_select(self):
        try:
            logger.info(f'Before :{self.training_data.columns}-->{self.training_data.shape}')
            logger.info(f'Before :{self.testing_data.columns}-->{self.testing_data.shape}')
            self.training_data,self.testing_data=FEATURE.feature_selecting(self.training_data,self.testing_data,self.y_train)
            logger.info(f'After :{self.training_data.columns}-->{self.training_data.shape}')
            logger.info(f'After :{self.testing_data.columns}-->{self.testing_data.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def data_balance(self):
        try:
            #logger.info(f'{self.y_train}')
            logger.info(f'\nBefore :{self.X_train_num}')
            logger.info(f'\nBefore :{self.X_test_num}')
            self.X_train_num,self.X_test_num=DATASCALE.data_scale(self.X_train_num,self.X_test_num)
            logger.info(f'\nAfter :{self.X_train_num}')
            logger.info(f'\nAfter :{self.X_test_num}')
            logger.info(f'================================================')
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            self.training_data=pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_data=pd.concat([self.X_test_num,self.X_test_cat],axis=1)
            logger.info(f'================================================')
            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")
            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def all_algo(self):
        try:
            logger.info(f'========================================================')
            logger.info(f'\n{self.training_data}')
            logger.info(f'\n{self.testing_data}')
            logger.info(f'\n{self.y_train}')
            logger.info(f'\n{self.y_test}')
            ALL_ALGO.all_algorithems(self.training_data,self.testing_data,self.y_train,self.y_test)
            logger.info(f'\n{self.training_data[:5]}')
            logger.info(f'\n{self.y_train[:5]}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == '__main__':
    try:
        path1='C:\\Users\\Ganesh\\OneDrive\\Desktop\\calories\\calories.csv'
        path2='C:\\Users\\Ganesh\\OneDrive\\Desktop\\calories\\exercise.csv'
        obj=CALORIES_BURNT_PREDICTION(path1,path2)
        obj.missing_values()
        obj.vt_outhand()
        obj.categori_num()
        #obj.feature_select()
        obj.data_balance()
        obj.all_algo()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
