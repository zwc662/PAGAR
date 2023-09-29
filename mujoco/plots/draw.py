import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a function to smooth the curves using a rolling average
def smooth_curve(df, window=150):
    #print(df.shape)
    wind = df.rolling(window=window, min_periods=1)
    mean = wind.mean()
    std = wind.std()
    return (mean, mean - std, mean + std)

def draw(title, types = ['gail'], label = None, limit = None, draw_legend = False):
    # Read the CSV files into a list of Pandas dataframes
    filenames = []
    labels = []
    for type in types:
        filenames = filenames + [f"run-.-tag-log_{title}_{type}_score.csv"]
        labels = labels + [f'{type.upper()}']    
        if 'gail' in type or 'vail' in type:
            filenames = filenames + [f"run-.-tag-log_{title}_p{type}_protagonist_score.csv"] #[f"run-.-tag-log_{title}_{type}_score.csv", f"run-.-tag-log_{title}_p{type}_antagonist_score.csv", f"run-.-tag-log_{title}_p{type}_protagonist_score.csv"]
            labels = labels + [f'protaognist_{type.upper()}']
    dfs = [pd.read_csv(filename) for filename in filenames]
    min_steps = 0 #limit
   
    for i, df in enumerate(dfs):
        print(labels[i])
        if 'IQ' in labels[i]:
            dfs[i]['Step'] = ((dfs[i]['Step'] - dfs[i]['Step'].iloc[0])/2048).astype('int')
        min_steps = min(min_steps, dfs[i]['Step'].max()) if min_steps is not None else dfs[i]['Step'].max()
    if limit is not None:
        min_steps = limit
    # Smooth each dataframe using the defined function
    smooth_dfs = [(df["Step"][df["Step"] <= min_steps], smooth_curve(df['Value'][df["Step"] <= min_steps])) for df in dfs]
    
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    print(min_steps)
    # Plot each smoothed dataframe on the axis
    batch = slice(None, min_steps, 1)
    for i, (x, (y_mean, y_low, y_high)) in enumerate(smooth_dfs):
        
        #x_std = df_std["Step"][df_mean["Step"] <= min_steps]
        ax.plot(x[batch], y_mean[batch], label= labels[i])
         
        ax.fill_between(x[batch], y_low[batch],  y_high[batch], alpha=0.2)


    if draw_legend:
        # Set the title and legend for the plot
        #ax.set_title(title)
        ax.legend(fontsize = 12, loc = 'best')
    plt.xlabel("iter")
    plt.ylabel("average return")
    plt.grid()
    # Display the plot
    plt.show()


titles = [
    
    #"InvertedPendulum-v2", 
    "Swimmer-v2", 
    #"Walker2d-v2", 
    #"HalfCheetah-v2",
    #"Hopper-v2",
    ]
types = [
    'gail',
    'vail'
]
limits = [
    1000,
    2000, #None,
    3000,
    3000,
    2000,
]


for idx, title in enumerate(titles): 
    def label(i):
        if i % 2 == 0:
            return '{type}'
        else:
            return 'protagonist_{type}'
    draw(title, ['gail', 'vail', 'iq'], label, limits[idx], True if idx == 0 else False)