import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

# --------Top of def main():--------
# ----Create initial data frames----
hr_df = pd.read_csv('HR_comma_sep.csv')
hr_df.rename(
    columns={
        'sales': 'department',
        'average_montly_hours': 'average_monthly_hours',
        'Work_accident': 'work_accident'},
    inplace=True)
# print(hr_df.groupby('department').count().index)
department_conversion = {
    'IT': 0, 'RandD': 1, 'accounting': 2, 'hr': 3,
    'management': 4, 'marketing': 5, 'product_mng': 6,
    'sales': 7, 'support': 8, 'technical': 9}
hr_df['department_number'] = hr_df['department'].apply(lambda x: department_conversion[x])
salary_conversion = {'low': 0, 'medium': 1, 'high': 2}
hr_df['salary_number'] = hr_df['salary'].apply(lambda x: salary_conversion[x])
left_df = hr_df[hr_df['left']==1]
stay_df = hr_df[hr_df['left']==0]
    

# ----Re-used Helper Functions----
def histogram_set(df, x_cols, y_sum_col, show=True):
    """Return None
    
    This function plots a matrix of histogram with a histogram chart for each
    column in x_cols, where each bar chart has bins for values in x_cols and
    bars with height equal to the number of elements in that bin.
    """
    width = min([len(x_cols), 3])
    height = math.ceil(len(x_cols) / 3)
    
    plt.figure(figsize=(13, 5))
    
    for i, x in enumerate(x_cols):
        if x:  # We can put place markers so that we can add data later
            plt.subplot(height, width, i+1)
            #sum_df = pd.DataFrame(df.groupby(x).sum()[y_sum_col])
            #plt.bar(list(sum_df.index), list(sum_df[y_sum_col]))
            plt.hist(df[x])
            plt.title('Total {} by {}'.format(y_sum_col, x))
            plt.xlabel(x)
            plt.ylabel('Total of {}'.format(y_sum_col))
    plt.tight_layout()
    
    if show:
        plt.show()
    
    
def plot_set_vs_set(df, x_cols, y_cols, legend_label=None, marker_size=2.0):
    """Return None
    
    This function plots a matrix of scatter plots with all combinations of
        the columns of df listed in x_cols, on the x-axis
    versus
        the columns of df listed in y_cols, on the y-axis
    There will be a row of scatter plots for each column listed in y_cols
    There will be a column of scatter plots for each column listed in x_cols
    """
    width = len(x_cols)
    height = len(y_cols)
    
    plt.figure(figsize=(13, 7))
    
    for i, y in enumerate(y_cols):
        for j, x in enumerate(x_cols):
            if x != y:
                plt.subplot(height, width, (i*width)+(j+1))
                plt.plot(
                    df[x], df[y], '*', markersize=marker_size,
                    label=legend_label, alpha=0.2)
                plt.title('{}\nvs. {}'.format(x, y))
                plt.xlabel(x)
                plt.ylabel(y)
                if legend_label and i==0 and j==0:
                    plt.legend(bbox_to_anchor=(0.13, -0.12))
    plt.tight_layout()
    plt.show()
    

# ----One Time Helper Functions----
def data_vis1(hr_df, left_df):
    labels = ('Left', 'Staying')
    count_left = len(left_df)
    count_stay = len(hr_df) - count_left
    sizes = [count_left, count_stay]
    colors = ['gold', 'yellowgreen']
    explode = (0.1, 0)  # explode 1st slice
     
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 3, 1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Percent of Employees That Left')
    
    # Distribution leavers' time spent at company
    plt.subplot(1, 3, 2)
    plt.hist(left_df['time_spend_company'])
    time_average = np.mean(left_df['time_spend_company'])
    plt.axvline(x=time_average, label='Average', color='red')
    plt.title('Distribution of Years Spent\n At Company Before Leaving')
    plt.xlabel('Years Spent At Company')
    plt.ylabel('Total That Left')
    plt.legend()
    
    # Distribution leavers' time spent at company
    plt.subplot(1, 3, 3)
    plt.hist(hr_df['time_spend_company'])
    time_average = np.mean(hr_df['time_spend_company'])
    plt.axvline(x=time_average, label='Average', color='red')
    plt.title('Distribution of Years\nSpent At Company')
    plt.xlabel('Years Spent At Company')
    plt.ylabel('Total Employees')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
def data_vis2(hr_df, left_df):
    # Plot quality employees that left
    histogram_set(
        left_df,
        ['last_evaluation', 'promotion_last_5years', 'salary_number'],
        'Employees That Left', show=False)
    
    # Set labels afterwards
    plt.subplot(1, 3, 1)
    plt.xlim(.36, 1.0)
    plt.subplot(1, 3, 2)
    plt.xticks(range(2), ('No', 'Yes'))
    plt.xlabel('Promotion in the Last 5 Years')
    plt.subplot(1, 3, 3)
    plt.xticks(range(3), ('Low', 'Medium', 'High'))
    plt.xlabel('Salary')
    
    plt.show()
    
    # Plot all quality employees
    histogram_set(
        hr_df,
        ['last_evaluation', 'promotion_last_5years', 'salary_number'],
        'All Employees', show=False)
    
    # Set labels afterwards
    plt.subplot(1, 3, 1)
    evaluation_average = np.mean(hr_df['last_evaluation'])
    plt.axvline(x=evaluation_average, label='Average', color='red')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.xticks(range(2), ('No', 'Yes'))
    plt.xlabel('Promotion in the Last 5 Years')
    plt.subplot(1, 3, 3)
    plt.xticks(range(3), ('Low', 'Medium', 'High'))
    plt.xlabel('Salary')
    
    plt.show()
    

def data_vis3(stay_df, left_df):
    x = [
        'number_project', 'average_monthly_hours',
        'time_spend_company', 'last_evaluation']
    y = ['satisfaction_level', 'last_evaluation']
    plot_set_vs_set(left_df, x, y, 'Left')
    
    
def data_vis4(stay_df, left_df):
    x = [
        'number_project', 'average_monthly_hours',
        'time_spend_company', 'last_evaluation']
    y = ['satisfaction_level', 'last_evaluation']
    plot_set_vs_set(stay_df, x, y, 'Stay')
    
    
def main():
    # For reference
    ['satisfaction_level', 'last_evaluation', 'number_project',
     'average_monthly_hours', 'time_spend_company', 'work_accident',
     'left', 'promotion_last_5years', 'department', 'salary']
    
    # ----Get an overview of turnover rate----
    # Percentage that left vs. stayed
    data_vis1(hr_df, left_df)
    
    
    # Draw histograms to find leaver correlations with other factors
    data_vis2(hr_df, left_df)
    
    # Plot scatter plots to find meaningful correlations
    data_vis3(stay_df, left_df)
    
    # Plot scatter plots to find meaningful correlations
    data_vis4(stay_df, left_df)
    

if __name__ == "__main__":
    main()