import seaborn as sns
import matplotlib.pyplot as plt


def set_plot_params(fig_size=(10, 5), label_size=12, font_size=11, 
                    font_type='Arial', label_weight='bold'):

    """
    Set default aesthetic parameters for all seaborn plots.
    """
            
    plt.style.use('fivethirtyeight')
    
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    plt.figure(figsize=fig_size)
    
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.family"] = font_type

    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["axes.labelweight"] = label_weight
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.xticks(rotation=45)



def plot_grouped_lineplot(df, grouping_col, time_var, target_var, 
                          title, y_title, x_title, fig_size=(10, 8), 
                          date_format=None, estimator='mean', *args, **kwargs):
    """
    Plot a grouped calculation as a lineplot in seaborn.
    """

    set_plot_params(fig_size)

    ax = sns.lineplot(x=time_var, y=target_var, data=df, hue=grouping_col, 
                      estimator=estimator, *args, **kwargs)

    ax.set(xlabel=x_title, ylabel=y_title)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, 
               frameon=False, borderaxespad=0.)

    if date_format:
        from matplotlib.dates import DateFormatter

        date_form = DateFormatter(date_format)
        ax.xaxis.set_major_formatter(date_form)
    
    plt.show()

        
def split_plot_by_group(df, split_col, n_plots, 
                        plotting_func, *args, **kwargs):
    """
    Allows you to easily split one plot into three plots according 
    to some grouping identifier. 
    """
    # default is "Top", "Middle", "Bottom"
    n_groups = sorted(list(df[split_col].unique()), reverse=True)
    
    for group in n_groups[:n_plots]:
        sub_df = df.query(f'{split_col} == "{group}"')
        plotting_func(df=sub_df, title=group, *args, **kwargs)


def plot_grouped_distribution(df, grouping_col, x_var, y_var, 
                              title, y_title, x_title, fig_size=(10, 8), 
                              plotting_func=sns.boxplot, *args, **kwargs):
    """
    Plot a grouped calculation as a violinplot in seaborn.
    """

    set_plot_params(fig_size)

    ax = plotting_func(x=x_var, y=y_var, hue=grouping_col, 
                       data=df, *args, **kwargs)

    ax.set(xlabel=x_title, ylabel=y_title)  
    plt.title(title)
     
    if grouping_col:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, 
                   frameon=False, borderaxespad=0.)
    
    plt.show()