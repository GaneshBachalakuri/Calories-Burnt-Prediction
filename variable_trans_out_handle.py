import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('variable_trans_out_handle')
from scipy import stats

class VAR_TRANS_OUT_HANDLE:
    def variable_transform_outliers(X_train, X_test):
        try:
            logger.info(f'{X_train.columns} ---> {X_train.shape}')
            logger.info(f'{X_test.columns} ---> {X_test.shape}')

            PLOT_PATH = "C:\\Users\\Ganesh\\OneDrive\\Desktop\\calories\\plots_path"
            # ================= BEFORE PLOTS =================
            for col in X_train.columns:
                plt.figure()
                X_train[col].plot(kind='kde', color='r')
                plt.title(f'KDE-{col}')
                plt.savefig(f'{PLOT_PATH}/kde_before_{col}.png')
                plt.close()

                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col}')
                plt.savefig(f'{PLOT_PATH}/boxplot_before_{col}.png')
                plt.close()

            # ================= TRANSFORMATION + CAPPING =================
            for col in X_train.columns:

                #Log transform (BEST for calories-like features)
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])

                #Quantile capping (from train only)
                lower = X_train[col].quantile(0.01)
                upper = X_train[col].quantile(0.99)

                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col] = X_test[col].clip(lower, upper)

            logger.info(f'After transform {X_train.shape}')
            logger.info(f'After transform {X_test.shape}')

            # ================= AFTER PLOTS =================
            for col in X_train.columns:
                plt.figure()
                X_train[col].plot(kind='kde', color='g')
                plt.title(f'KDE-{col}')
                plt.savefig(f'{PLOT_PATH}/kde_after_{col}.png')
                plt.close()

                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col}')
                plt.savefig(f'{PLOT_PATH}/boxplot_after_{col}.png')
                plt.close()

            return X_train, X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in line {error_line.tb_lineno}: {error_msg}')
