import pandas as pd
import matplotlib as plt
import seaborn as sns

def plot_col_hist(data:pd.DataFrame) -> None:
    '''
    Goes through columns and plots histograms
    '''
    columns_list = data.columns
    for i in columns_list:
        data[i].plot(kind='hist', bins=100, figsize=(14, 6),  title=f"Histogram of {i}")
        plt.show()

    return


def analyze_correlation(data:pd.DataFrame, img_name:str='correlation_matrix') -> pd.DataFrame:
    '''
    Makes correlation matrix and plots histogram
    '''
    correlation_matrix = data.corr()

    plt.figure(figsize=(10,10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig(f'plots/{img_name}.png')
    plt.show()

    return correlation_matrix
