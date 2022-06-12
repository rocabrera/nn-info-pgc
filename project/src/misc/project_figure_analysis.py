import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')

def set_scatterplot_legend(ax):
    ax.set_xlabel("I(X,T)", fontsize=12)
    ax.set_ylabel("I(Y,T)", fontsize=12)

    leg = ax.legend(loc='lower right', 
                    title='Epochs',
                    fontsize=12)
    # make opaque legend
    for lh in leg.legendHandles:
        fc_arr = lh.get_fc().copy()
        fc_arr[:, -1] = 1
        lh.set_fc(fc_arr)
        lh.set_edgecolor("black")
        lh.set_linewidth(1.4)        
        lh.set_alpha(1)
    last_element_legend = leg.get_texts()[-1]
    last_element_legend.set_text(fr"$\geq$ {last_element_legend.get_text()}")
    return leg

def normalize_dataset(df):
    
    scaler = MinMaxScaler()
    mutual_information_columns = ["I(X,T)", "I(Y,T)"]
    scaled_feats = scaler.fit_transform(df[mutual_information_columns])
    return (df.assign(**{feat_name:new_col for feat_name, new_col in zip(mutual_information_columns, scaled_feats.T)})
               .round(3)  
    )
    


def create_lineplot_rand_init(df):
    scaler = LogNorm()
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    cmap.colors = scaler(cmap.colors)


    g = sns.FacetGrid(normalize_dataset(df), col="rand_init", height=3.5, col_wrap=5)
    g.map_dataframe(sns.lineplot,
                    x="I(X,T)", 
                    y="I(Y,T)",
                    hue="epoch", 
                    estimator=None,
                    lw=0.5,
                    marker="o",
                    palette=cmap,
                    legend=None)
    plt.show()

def create_scatter_mean_trajectory(df):
    
    scaler = LogNorm()
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    cmap.colors = scaler(cmap.colors)

    fig, ax = plt.subplots(figsize = (6,4))

    new_dataset = (normalize_dataset(df).groupby(["epoch", "layer"], as_index=False).mean())
    g = sns.scatterplot(data=new_dataset, 
                        x='I(X,T)', 
                        y="I(Y,T)", 
                        hue="epoch",
                        alpha=0.3,
                        edgecolor="black",
                        palette=cmap,
                        linewidth=1.4,
                        cmap=sns.cubehelix_palette(as_cmap=True),
                        ax=ax)

    leg = set_scatterplot_legend(g)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.show()