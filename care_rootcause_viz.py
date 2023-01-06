import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_rgb
import seaborn as sns

## compute the ranks of the causal path
def ranking():
    raw = pd.read_csv('result/care_ace_result.csv', names=['Path', 'ACE', 'Ql', 'Qu'])
    data = [raw['Path'], abs(raw['ACE']), abs(raw['Ql']), abs(raw['Qu'])]
    df = pd.DataFrame(data)
    df = df.T
    rank_path = df.sort_values(by=['ACE'], ascending=False)
    N = [df['ACE'], df['Ql'], df['Qu']]
    norm = preprocessing.scale(N)
    post = pd.DataFrame({'Path':raw['Path'], 'ACE':norm[0], 'Ql':norm[1], 'Qu':norm[2]})
    rank_path.to_csv('result/rank_path.csv', index=False)
    post.to_csv('result/cluster.csv', index=False, header=False)

def energy_ranks():
    df = pd.read_csv('result/cluster_energy.csv', names=['ACE'])
    # k means
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['Rank'] = kmeans.fit_predict(df[['ACE']])
    # get centroids
    centroids = kmeans.cluster_centers_
    cen_y = [i[0] for i in centroids] 
    ## add to df
    df['cen_y'] = df.Rank.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3]})
    colors = ['b', 'r', 'g', 'm']
    df['c'] = df.Rank.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]})  

    x = range(1, 147)
    plt.rcParams.update({'figure.figsize':(4,7.1)})
    plt.scatter(x, df.ACE, c=df.c, alpha = 0.5, s=200)
    colors = ['r', 'm', 'b', 'g'] # sorted
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Rank {}'.format(i+1), 
                markerfacecolor=mcolor, markersize=20) for i, mcolor in enumerate(colors)]
    # plot Defense mean
    plt.xlim(-30, 160)
    fontsize = 30
    labelsize = 26
    plt.xlabel('Causal Paths', fontsize=fontsize)
    plt.xticks([])
    plt.ylabel(r'$z-score\ (\mathrm{\mathbb{P}_{ACE}})$', fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(handles=legend_elements, fontsize=20, loc='center left', framealpha=0.8)
    plt.gca().yaxis.grid(True, alpha=0.5)
    plt.savefig('fig/rank_paths_energy_RAL.pdf', dpi=100, bbox_inches='tight')    

def MS_rank():
    df = pd.read_csv('result/cluster_mission.csv', names=['ACE'])
    # k means
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['Rank'] = kmeans.fit_predict(df[['ACE']])
    # get centroids
    centroids = kmeans.cluster_centers_
    cen_y = [i[0] for i in centroids] 
    ## add to df
    df['cen_y'] = df.Rank.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3]})
    colors = ['r', 'g', 'b', 'm']
    df['c'] = df.Rank.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3]}) 

    x = range(1, 126)
    plt.rcParams.update({'figure.figsize':(4,7.1)})
    plt.scatter(x, df.ACE, c=df.c, alpha = 0.5, s=200)
    colors = ['r', 'm', 'b', 'g'] # sorted
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Rank {}'.format(i+1), 
                markerfacecolor=mcolor, markersize=20) for i, mcolor in enumerate(colors)]
    # plot Defense mean
    plt.xlim(-30, 160)
    fontsize = 30
    labelsize = 26
    plt.xlabel('Causal Paths', fontsize=fontsize)
    plt.xticks([])
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(handles=legend_elements, fontsize=22, loc='center left', framealpha=0.8)
    plt.gca().yaxis.grid(True, alpha=0.5)
    plt.savefig('fig/rank_paths_mission_RAL.pdf', dpi=100, bbox_inches='tight')      

## -----------------------------------------------------------------------------       

### validating ranks
def energy():
    rank_e = pd.read_csv("result/exp/energy.csv")
    plt.rcParams.update({'figure.figsize':(3.7,2.9)})
    v = sns.violinplot(data= rank_e, x='Rank', y='Energy (Whr)', hue='Husky', split=True,
                        inner='box', bw=.4, width=1, linewidth=2.5)                  
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    v.legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], labels=['Simulator','Physical'], fontsize=14, title_fontsize=14,
            title="Husky", handlelength=3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    fontsize = 18
    labelsize = 18
    v.set(xlabel=None)
    plt.ylabel("Energy (Whr)", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.savefig('fig/rank_energy.pdf', dpi=100, bbox_inches='tight') 

def energy_td():
    rank_e = pd.read_csv("result/exp/energy.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.4)})
    v = sns.violinplot(data= rank_e, x='Rank', y='Traveled_distance', hue='Husky', split=True,
                        inner='box', bw=.4, width=0.8, linewidth=2.5)
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    v.legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], labels=['Simulator','Physical'], fontsize=14,
            title="Husky", handlelength=3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel(r"TD $(meters)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.legend([],[], frameon=False)
    plt.savefig('fig/rank_energy_td.pdf', dpi=100, bbox_inches='tight') 

def energy_RP():
    rank_e = pd.read_csv("result/exp/energy.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.7)})
    v = sns.violinplot(data= rank_e, x='Rank', y='DWA_new_plan', hue='Husky', split=True,
                        bw=.4, width=0.8, linewidth=2.5)
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel(r"RP $(count)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.legend([],[], frameon=False)
    plt.savefig('fig/rank_energy_RP.pdf', dpi=100, bbox_inches='tight') 

def energy_RE():
    rank_e = pd.read_csv("result/exp/energy.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.4)})
    v = sns.violinplot(data= rank_e, x='Rank', y='Recovery_executed', hue='Husky', split=True,
                        bw=.4, width=0.8, linewidth=2.5)
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel(r"RE $(count)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.legend([],[], frameon=False)
    plt.savefig('fig/rank_energy_re.pdf', dpi=100, bbox_inches='tight')  

def mission_success():
    rank_e = pd.read_csv("result/exp/mission_proba_sim.csv")
    rank_e_phy = pd.read_csv("result/exp/mission_proba_real.csv")

    plt.rcParams.update({'figure.figsize':(3.5,2.5)})
    x = range(1,51)
    x_phy = range(1,11)

    plt.plot(x, rank_e['Rank 1'], alpha=0.5, color='r', drawstyle="steps", label=r'$Rank_1$ sim.')
    plt.fill_between(x, rank_e['Rank 1'], step="pre", color='r', alpha=0.5)
    plt.plot(x, rank_e['Rank 2'], alpha=0.4, color='b', drawstyle="steps", label=r'$Rank_3$ sim.')
    plt.fill_between(x, rank_e['Rank 2'], step="pre", color='b', alpha=0.5)
    plt.plot(x, rank_e['Rank 3'], alpha=0.5, color='g', drawstyle="steps", label=r'$Rank_4$ sim.')
    plt.fill_between(x, rank_e['Rank 3'], step="pre", color='g', alpha=0.5)

    plt.plot(x_phy, rank_e_phy['Rank 1'], alpha=0.5, color='r',  ls='--',  label=r'$Rank_1$ phy.')
    plt.fill_between(x_phy, rank_e_phy['Rank 1'],  color='r', alpha=0.2)
    plt.plot(x_phy, rank_e_phy['Rank 2'], alpha=0.4, color='b', ls='--', label=r'$Rank_3$ phy.')
    plt.fill_between(x_phy, rank_e_phy['Rank 2'],  color='b', alpha=0.2)
    plt.plot(x_phy, rank_e_phy['Rank 3'], alpha=0.5, color='g', ls='--', label=r'$Rank_4$ phy.')
    plt.fill_between(x_phy, rank_e_phy['Rank 3'], color='g', alpha=0.2)

    fontsize = 18
    labelsize = 18
    plt.xlabel(r"Number of mission success ($x_m$)", fontsize=fontsize)
    plt.xlim(0, 30)
    plt.ylim(ymin=0)
    plt.ylabel(r"$P(X=x_m)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(title='Husky', title_fontsize=14,fontsize=14, bbox_to_anchor=(0.8, 0.6), loc="center left", borderaxespad=0, ncol=1)
    # plt.setp(v.collections, alpha=.5)
    plt.xlim(left=0)
    plt.savefig('fig/exp_rank_m_RAL.pdf', dpi=100, bbox_inches='tight')        

def MS_rns():
    rank_ms = pd.read_csv("result/exp/mission_success.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.5)})

    v = sns.violinplot(data= rank_ms, x='Rank', y='RNS', hue='Husky', split=True,
                        inner='box', bw=.4, width=1, linewidth=2.3, palette=['r', 'b', 'g'])
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel("RNS", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend([],[], frameon=False)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.savefig('fig/rank_ms_rns.pdf', dpi=100, bbox_inches='tight')  

def MS_re():
    rank_ms = pd.read_csv("result/exp/mission_success.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.65)})
    v = sns.violinplot(data= rank_ms, x='Rank', y='Recovery_executed', hue='Husky', split=True,
                        inner='box', bw=.4, width=1, linewidth=2.3, palette=['r', 'b', 'g'])
    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    v.legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], labels=['Simulator','Physical'], fontsize=16,
            title="Husky", handlelength=3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel(r"RE $(count)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend([],[], frameon=False)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.savefig('fig/rank_ms_re.pdf', dpi=100, bbox_inches='tight')  

def MS_erg():
    rank_ms = pd.read_csv("result/exp/mission_success.csv")
    plt.rcParams.update({'figure.figsize':(3.5,3.4)})
    v = sns.violinplot(data= rank_ms, x='Rank', y='Error_rotating_goal', hue='Husky', split=True,
                        inner='box', bw=.4, width=1, linewidth=2.3, palette=['r', 'b', 'g'])

    colors = ['crimson', 'dodgerblue', 'limegreen']
    handles = []
    for ind, violin in enumerate(v.findobj(PolyCollection)):
        rgb = to_rgb(colors[ind // 2])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
        violin.set_facecolor(rgb)
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))
    v.legend(handles=[tuple(handles[::2]), tuple(handles[1::2])], labels=['Simulator','Physical'], fontsize=16,
            title="Husky", handlelength=3, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    fontsize = 20
    labelsize = 18.5
    v.set(xlabel=None)
    plt.ylabel(r"ERG $(count)$", fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend([],[], frameon=False)
    sns.set_style("ticks",{'axes.grid' : True})
    plt.savefig('fig/rank_ms_erg.pdf', dpi=100, bbox_inches='tight')   

## -----------------------------------------------------------------------------       

