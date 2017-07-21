import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
from scipy.stats import ttest_ind

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
hr_df['department_number'] = hr_df['department'].apply(
    lambda x: department_conversion[x])
salary_conversion = {'low': 0, 'medium': 1, 'high': 2}
hr_df['salary_number'] = hr_df['salary'].apply(lambda x: salary_conversion[x])
left_df = hr_df[hr_df['left']==1]
stay_df = hr_df[hr_df['left']==0]
    

# ----Re-used Helper Functions----
def histogram_set(
        df, x_cols, y_sum_col, overlay=False, show=True, bin_list=None,
        tansparency=1.0, label=None):
    """Return None
    
    This function plots a matrix of histograms, with a histogram chart for
    each column in x_cols, where each bar chart has bins for values in x_cols
    and bars with height equal to the number of elements in that bin. The
    histograms are normalized so that the heights of their columns sum to 1
    (as opposed to their integral over the bins summing to 1).
    """
    width = min([len(x_cols), 3])
    height = math.ceil(len(x_cols) / 3)
    
    if (not overlay):
        plt.figure(figsize=(13, 5))
    
    for i, x in enumerate(x_cols):
        if x:  # We can put place markers so that we can add data later
            plt.subplot(height, width, i+1)
            normalized_weights = (np.ones_like(df[x])
                / float(len(df[x])))
            bins = None
            if bin_list:
                bins = bin_list[i]
            plt.hist(
                df[x], bins=bins, weights=normalized_weights, alpha=0.5,
                label=label)
            plt.title('Total {} by {}'.format(y_sum_col, x))
            plt.xlabel(x)
            plt.ylabel('Total of {}'.format(y_sum_col))
    plt.tight_layout()
    
    if show:
        plt.show()
    
    
def scatter_set_vs_set(
        df, x_cols, y_cols, legend_label=None, marker_size=2.0,
        overlay=False, show=True, transparency=1.0):
    """Return None
    
    This function plots a matrix of scatter plots with all combinations of
        the columns of df listed in x_cols, on the x-axis
    versus
        the columns of df listed in y_cols, on the y-axis
    There will be a row of scatter plots for each column listed in y_cols
    There will be a column of scatter plots for each column listed in x_cols
    """
    # Dimensions of subplots
    width = len(x_cols)
    height = len(y_cols)
    
    if (not overlay):
        plt.figure(figsize=(13, 7))
    
    for i, y in enumerate(y_cols):
        for j, x in enumerate(x_cols):
            if x != y:
                plt.subplot(height, width, (i*width)+(j+1))
                plt.plot(
                    df[x], df[y], '*', markersize=marker_size,
                    label=legend_label, alpha=transparency)
                plt.title('{}\nvs. {}'.format(x, y))
                plt.xlabel(x)
                plt.ylabel(y)
                if legend_label and i==0 and j==0:
                    plt.legend()
    plt.tight_layout()
    
    if show:
        plt.show()


def boxplot_set_vs_set(
        df, df_bins, x_cols, y_cols, legend_label=None, overlay=False,
        show=True, color=None, line_width=None):
    """Return None
    
    This function plots a matrix of box plots with all combinations of
        the columns of df listed in x_cols, on the x-axis
    versus
        the columns of df listed in y_cols, on the y-axis
    There will be a row of scatter plots for each column listed in y_cols
    There will be a column of scatter plots for each column listed in x_cols
    """
    # Dimensions of subplots
    width = len(x_cols)
    height = len(y_cols)
        
    if (not overlay):
        plt.figure(figsize=(13, 7))
        
    for i, y in enumerate(y_cols):
        for j, x in enumerate(x_cols):
            if x == y:
                plt.subplot(height, width, (i * width) + (j + 1))
                plt.axis('off')
                break
            subplot_data = []
            bin_names = ['']
            unique_values = df_bins[x].unique()
            unique_values.sort()
            
            # If there are <= 10 x-values, use them as bins for each box plot
            if len(unique_values) <= 10:
                # Get the data for each bin's boxplot
                for unique_value in unique_values:
                    subplot_data.append(df[df[x]==unique_value][y])
                    bin_names.append(unique_value)
                    
            # Else separate x-values into bins
            else:
                bin_size = (max(unique_values) - min(unique_values)) / 10
                bin_start = min(unique_values)
                bin_end = bin_start + bin_size
                
                # Get the data for each bin's boxplot
                for count in range(10):
                    if count == 0:
                        subplot_data.append(
                            df[(bin_start<=df[x]) & (df[x]<=bin_end)][y])
                        bin_names.append(
                            "{:02.2f} - {:02.2f}".format(bin_start, bin_end))
                        #bin_names.append(
                        #   "{} <= x <= {}".format(bin_start, bin_end))
                    else:
                        subplot_data.append(
                            df[(bin_start<df[x]) & (df[x]<=bin_end)][y])
                        bin_names.append(
                            "{:02.2f} - {:02.2f}".format(bin_start, bin_end))
                        #bin_names.append(
                        #   "{} < x <= {}".format(bin_start, bin_end))
                    bin_start = bin_end
                    bin_end += bin_size
            # Draw the boxplots with descriptors
            plt.subplot(height, width, (i * width) + (j + 1))
            bp = plt.boxplot(
                subplot_data, labels=([legend_label] * (len(bin_names) - 1)))
            if line_width:
                for part in bp.keys():
                    plt.setp(bp[part], linewidth=line_width)
            if color:
                for part in bp.keys():
                    plt.setp(bp[part], color=color)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.xticks(
                range(len(bin_names)), bin_names, rotation='vertical')
            plt.tight_layout()
            plt.title('{}\nvs. {}'.format(x, y))
    if show:
        plt.show()


# ----One Time Helper Functions----
def data_vis1(
        hr_df, left_df, labels=('Left', 'Staying'),
        colors = ['gold', 'yellowgreen'], bins=range(0, 11)):
    count_left = len(left_df)
    count_stay = len(hr_df) - count_left
    sizes = [count_left, count_stay]
    explode = (0.1, 0)  # explode 1st slice
     
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 3, 1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Percent of Employees That Left')
    
    # Distribution leavers' time spent at company
    plt.subplot(1, 3, 2)
    # Simply setting normed=True makes the integral of the histogram = 1,
    # not cumulative heights, so we use weights instead
    normalized_weights = (np.ones_like(left_df['time_spend_company'])
        / float(len(left_df['time_spend_company'])))  
    plt.hist(
        left_df['time_spend_company'],
        weights=normalized_weights, bins=bins)
    time_average = np.mean(left_df['time_spend_company'])
    plt.axvline(x=time_average, label='Average', color='red')
    plt.text(
        time_average, 0.3, "x={:04.2f}".format(time_average))
    plt.title('Distribution of Years Spent\n At Company, If Left')
    plt.ylim([0, 0.6])
    plt.xlabel('Years Spent At Company')
    plt.ylabel('Percent Employees')
    plt.legend()
    
    # Distribution leavers' time spent at company
    plt.subplot(1, 3, 3)
    normalized_weights = (np.ones_like(hr_df['time_spend_company'])
        / float(len(hr_df['time_spend_company'])))
    plt.hist(
        hr_df['time_spend_company'],
        weights=normalized_weights, bins=bins)
    time_average = np.mean(hr_df['time_spend_company'])
    plt.axvline(x=time_average, label='Average', color='red')
    plt.text(
        time_average, 0.3, "x={:03.2f}".format(time_average))
    plt.title('General Distribution of Years\nSpent At Company')
    plt.ylim([0, 0.6])
    plt.xlabel('Years Spent At Company')
    plt.ylabel('Percent Employees')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(ttest_ind(left_df['time_spend_company'], hr_df['time_spend_company'], equal_var=False))
    
    
def data_vis2(hr_df, left_df):
    # Plot quality employees that left
    evaluation_bins = np.multiply(0.1, range(0,11))
    subplot_bins = [evaluation_bins, None, None]
    histogram_set(
        left_df,
        ['last_evaluation', 'promotion_last_5years', 'salary_number'],
        'Employees', show=False, tansparency=0.6,
        bin_list=subplot_bins, label='Employees That Left')
    
    # Overlay all quality employees
    histogram_set(
        hr_df,
        ['last_evaluation', 'promotion_last_5years', 'salary_number'],
        'Employees', overlay=True, show=False, tansparency=0.6,
        bin_list=subplot_bins, label='All Employees')
    
    # Set labels afterwards
    plt.subplot(1, 3, 2)
    plt.xticks(range(2), ('No', 'Yes'))
    plt.xlabel('Promotion in the Last 5 Years')
    plt.subplot(1, 3, 3)
    plt.xticks(range(3), ('Low', 'Medium', 'High'))
    plt.xlabel('Salary')
    plt.legend(loc='upper right')
    
    plt.show()
    

def data_vis3(stay_df, left_df):
    x_values = [
        'number_project', 'average_monthly_hours',
        'time_spend_company', 'last_evaluation']
    y_values = ['satisfaction_level', 'last_evaluation']
    scatter_set_vs_set(
        left_df, x_values, y_values, legend_label='Left', show=False,
        transparency=0.5, marker_size=4)
    
    scatter_set_vs_set(
        stay_df, x_values, y_values, legend_label='Stay', overlay=True,
        transparency=0.1, marker_size=2)
    
    
def data_vis4(stay_df, left_df, hr_df):
    x_values = [
        'number_project', 'average_monthly_hours',
        'time_spend_company', 'last_evaluation']
    y_values = ['satisfaction_level']
    boxplot_set_vs_set(
        stay_df, hr_df, x_values, y_values, legend_label='Stay',
        show=False, color='red', line_width=1.5)
    
    boxplot_set_vs_set(
        left_df, hr_df, x_values, y_values, legend_label='Left',
        overlay=True, color='blue')
    
    x_values = [
        'number_project', 'average_monthly_hours',
        'time_spend_company', 'last_evaluation']
    y_values = ['last_evaluation']
    boxplot_set_vs_set(
        stay_df, hr_df, x_values, y_values, legend_label='Stay',
        show=False, color='red', line_width=1.5)
    
    boxplot_set_vs_set(
        left_df, hr_df, x_values, y_values, legend_label='Left',
        show=False, overlay=True, color='blue')
    
    # Draw legend
    red_line = mlines.Line2D([], [], color='blue',
        markersize=15, label='Employees That Left')
    blue_line = mlines.Line2D([], [], color='red',
        markersize=15, label='Employees That Did Not Leave')
    handles = [red_line, blue_line]
    labels = [h.get_label() for h in handles]
    plt.legend(handles=handles, labels=labels) 
    plt.show()
    
    
def main():
    # ----For reference----
    
    ['satisfaction_level', 'last_evaluation', 'number_project',
     'average_monthly_hours', 'time_spend_company', 'work_accident',
     'left', 'promotion_last_5years', 'department', 'salary']
    
    # ----Are many employees actually leaving?----
    # Percentage that left vs. stayed
    data_vis1(hr_df, left_df)
    
    # ----What type of people are leaving?----
    # Draw histograms to find leaver correlations with other factors
    data_vis2(hr_df, left_df)
    
    # ----Correlate data rangers to leavers and non-leavers----
    # Plot scatter plots to find meaningful correlations
    data_vis3(stay_df, left_df)
    
    # Plot scatter plots to find meaningful correlations
    data_vis4(stay_df, left_df, hr_df)
    

if __name__ == "__main__":
    main()