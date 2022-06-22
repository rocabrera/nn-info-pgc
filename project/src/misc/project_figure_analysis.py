import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')

def set_scatterplot_legend(ax):
    ax.set_xlabel("I(X,T)", fontsize=12)
    ax.set_ylabel("I(Y,T)", fontsize=12)

    leg = ax.legend(loc='lower right', 
                    title='Épocas',
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
    
def rename_experiment_columns(df):
    
    return df.rename(columns={
        "epoch": "Épocas",
        "rand_init": "Inicialização",
        "layer": "Camadas",
        "valid_auc": "Acurácia - validação",
        "train_auc": "Acurácia - treino",
        "valid_loss": "Loss - validação",
        "train_loss": "Loss - treino",
    })
    


def create_lineplot_rand_init(df, savepath=None):
    scaler = LogNorm()
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    cmap.colors = scaler(cmap.colors)


    g = sns.FacetGrid(df.pipe(normalize_dataset).pipe(rename_experiment_columns), col="Inicialização", height=3.5, col_wrap=5)
    g.map_dataframe(sns.lineplot,
                    x="I(X,T)", 
                    y="I(Y,T)",
                    hue="Épocas", 
                    estimator=None,
                    lw=0.5,
                    marker="o",
                    palette=cmap)
    
    
    ax = g.axes.ravel()[-1]
    ax.legend()
    leg = ax.legend(loc='lower right', 
                bbox_to_anchor=(0.5, .8, 1., .102),
                title='Épocas',
                fontsize=12)
    # make opaque legend
    last_element_legend = leg.get_texts()[-1]
    last_element_legend.set_text(fr"$\geq$ {last_element_legend.get_text()}")
    if savepath:
        _, ext = os.path.splitext(savepath)
        plt.savefig(savepath, format="svg", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_scatter_mean_trajectory(df, savepath=None):
    
    scaler = LogNorm()
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    cmap.colors = scaler(cmap.colors)

    fig, ax = plt.subplots(figsize = (6,4))

    new_dataset = (df.pipe(normalize_dataset).pipe(rename_experiment_columns).groupby(["Épocas", "Camadas"], as_index=False).mean())
    g = sns.scatterplot(data=new_dataset, 
                        x='I(X,T)', 
                        y="I(Y,T)", 
                        hue="Épocas",
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
    if savepath:
        _, ext = os.path.splitext(savepath)
        plt.savefig(savepath, format="svg")
        plt.close()
    else:
        plt.show()
        
def create_weights_per_epoch(df, savepath=None):
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    var_name = "Métrica"
    value_name = "Valor"
    value_vars = (r"$|\overline{W}|$", r"$\sigma_W$")
    
    aux = (df.pipe(normalize_dataset)
             .pipe(rename_experiment_columns)
             .abs()
             .rename(columns = {"mean_weight_layer": r"$|\overline{W}|$",
                                "std_weight_layer":  r"$\sigma_W$"
                               })
             .groupby(["Épocas", "Camadas"], as_index=False)
             .mean()
             .melt(id_vars=["Épocas", "Camadas"],
                   value_vars=value_vars,
                   value_name=value_name,
                   var_name=var_name)
          )


    sns.lineplot(data=aux, x="Épocas", y=value_name, hue="Camadas", style=var_name, palette="Set2", linewidth=2, ax=ax)
        
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if savepath:
        _, ext = os.path.splitext(savepath)
        plt.savefig(savepath, format="svg")
        plt.close()
    else:
        plt.show()
        
def create_information_per_epoch(df, savepath=None):
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    var_name = "Informação Mútua"
    value_name = "Valor"
    value_vars = "I(X,T)", "I(Y,T)"
    
    aux = (df.pipe(normalize_dataset)
             .pipe(rename_experiment_columns)
             .abs()
             .groupby(["Épocas", "Camadas"], as_index=False)
             .mean()
             .melt(id_vars=["Épocas", "Camadas"],
                   value_vars=value_vars,
                   value_name=value_name,
                   var_name=var_name)
          )


    sns.lineplot(data=aux, x="Épocas", y=value_name, hue="Camadas", style=var_name, palette="Set2", linewidth=2, ax=ax)
        
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if savepath:
        _, ext = os.path.splitext(savepath)
        plt.savefig(savepath, format="svg")
        plt.close()
    else:
        plt.show()
        
def create_performance_per_epoch(df, savepath=None):
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    var_name = "Desempenho"
    value_name = "Valor"
    value_vars = ("Acurácia - validação", "Loss - validação")
    
    aux = (df.pipe(normalize_dataset)
             .pipe(rename_experiment_columns)
             .abs()
             .groupby(["Épocas"], as_index=False)
             .mean()
             .melt(id_vars=["Épocas"],
                   value_vars=value_vars,
                   value_name=value_name,
                   var_name=var_name)
          )


    sns.lineplot(data=aux, x="Épocas", y=value_name, hue=var_name, palette="Set2", linewidth=2, ax=ax)
        
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if savepath:
        _, ext = os.path.splitext(savepath)
        plt.savefig(savepath, format="svg")
        plt.close()
    else:
        plt.show()
