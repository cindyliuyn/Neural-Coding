import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

pid = pd.read_csv('pid_results/all_pid.csv')
pid = pid[pid['redun_s1_s2'] > 0]

def get_hist_plot(df, col_names, colors_, fig_size, file_name):
    """
    Plot the histograms of the PID components
    """
    fig, ax = plt.subplots(figsize=fig_size)
    for col_name, color_ in zip(col_names, colors_.copy()):
        label = col_name + ' $\mu$ = ' + str(round(df[col_name].mean(), 3)) \
                + ' $\sigma$ = ' + str(round(df[col_name].std(), 3))
        sns.distplot(df[col_name], ax=ax, color=color_, label=label)
    ax.set_xlabel('Information (bits)')
    ax.grid(True)
    ax.set_title('')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'pid_results/{file_name}')
    plt.show()

get_hist_plot(pid, ['syn_s1_s2', 'uniq_s1', 'redun_s1_s2'], ['#FF5733', '#33B9FF', '#BAA16E'], (6, 6), 'syn_uniq_redun_hist.png')


def calculate_info_contributions(df):
    contribution_percentage = df[['uniq_s1','uniq_s2','redun_s1_s2','syn_s1_s2']]\
    .pipe(lambda df: df.apply(lambda row: row/row.sum(), axis=1))

    print('Mean contribution % of each component to the total information:')
    print(contribution_percentage.mean() * 100)

calculate_info_contributions(pid)

def visualise_info_by_timestep(df):
    """
    Lineplot of time step vs information components
    """
    df_grouped = df.groupby('time_step')
    df_grouped.mean().drop(columns=['source_neuron_1', 'source_neuron_2', 'neuron_distance'])\
        .plot(subplots=False, figsize=(10, 5))
    plt.savefig('pid_results/info_vs_time.png')
    plt.show()

visualise_info_by_timestep(pid)

def visualise_info_by_overlapping(df, col_name, title_name):
    """
    Lineplot of distance bucket vs information components
    """
    num_of_buckets = 25

    df_grouped = df.assign(nd_buk = lambda x: pd.cut(x['overlapping'], num_of_buckets)).groupby('nd_buk')
    df_grouped_mean = df_grouped.mean() # 'neuron_distance'

    all_cols = list(df_grouped_mean.columns)
    all_cols.remove(col_name)
    overlapping_axis = [round((interval.left + interval.right) / 2, 3) for interval in df_grouped_mean.index]

    corr, p_value = pearsonr(np.array(overlapping_axis), np.array(df_grouped_mean[col_name]))
    print(f'correlation: {corr}, p-value: {p_value}')
    plt.scatter(np.array(overlapping_axis), np.array(df_grouped_mean[col_name]))

    data = np.hstack((np.array(overlapping_axis).reshape(-1, 1), np.array(df_grouped_mean[col_name]).reshape(-1, 1)))
    np.savetxt(f'pid_results/{col_name}_data.csv', data, delimiter=',')

    plt.xlabel('Receptive Field Overlapping Area')
    plt.ylabel('Information (bits)')
    plt.title(title_name)
    plt.savefig(f'pid_results/{col_name}_vs_distance.png')
    plt.show()


visualise_info_by_overlapping(pid, 'syn_s1_s2', 'Synergistic Information')
visualise_info_by_overlapping(pid, 'uniq_s1', 'Unique Information')
visualise_info_by_overlapping(pid, 'redun_s1_s2', 'Redundant Information')


# T-test to check that is synergistic information significantly different between the two buckets of neuron distance?
def calculate_ttest(df, info_component):
    num_of_buckets = 25
    df_grouped = df.assign(nd_buk=lambda x: pd.cut(x['overlapping'], num_of_buckets)).groupby('nd_buk')
    syn_by_distance = [i for i in df_grouped[info_component]]
    for i in range(len(syn_by_distance)):
        print(f'bucket 1 and {i+1}: {ttest_ind(syn_by_distance[0][1], syn_by_distance[i][1])}')

calculate_ttest(pid, 'syn_s1_s2')
calculate_ttest(pid, 'redun_s1_s2')

