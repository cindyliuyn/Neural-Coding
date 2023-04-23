import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

pid = pd.read_csv('pid_results/all_pid.csv')

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

filtered_pid = pid[pid['redun_s1_s2'] > 0]
get_hist_plot(filtered_pid, ['syn_s1_s2', 'uniq_s1'], ['#FF5733', '#33B9FF'], (6, 6), 'syn_uniq_hist.png')
get_hist_plot(filtered_pid, ['redun_s1_s2'], ['#BAA16E'], (5, 5), 'redun_hist.png')


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


def visualise_info_by_neuron_distance(df):
    """
    Lineplot of distance bucket vs information components
    """
    num_of_buckets = 20
    df_grouped = df.assign(nd_buk = lambda x: pd.cut(x['neuron_distance'], num_of_buckets)).groupby('nd_buk')
    df_grouped.mean().drop(columns=['source_neuron_1', 'source_neuron_2', 'time_step', 'neuron_distance'])\
    .plot(subplots=False, figsize=(10, 5))  # [['syn_s1_s2']]
    plt.savefig('pid_results/info_vs_distance.png')
    plt.show()


visualise_info_by_neuron_distance(pid)


# T-test to check that is synergistic information significantly different between the two buckets of neuron distance?
def calculate_ttest(df, info_component):
    num_of_buckets = 10
    df_grouped = df.assign(nd_buk=lambda x: pd.cut(x['neuron_distance'], num_of_buckets)).groupby('nd_buk')
    syn_by_distance = [i for i in df_grouped[info_component]]
    for i in range(len(syn_by_distance)):
        print(f'bucket 1 and {i+1}: {ttest_ind(syn_by_distance[0][1], syn_by_distance[i][1])}')

calculate_ttest(pid, 'syn_s1_s2')
calculate_ttest(pid, 'redun_s1_s2')


# Calculate Correlation between time step and information
pid.groupby('time_step').corr().loc[(slice(None), 'neuron_distance'), :]