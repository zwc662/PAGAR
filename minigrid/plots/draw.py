import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(df, window=30):
    #print(df.shape)
    wind = df.rolling(window=window, min_periods=1)
    mean = wind.mean()
    std = wind.std()
    return (mean, mean - std, mean + std)

def draw(title, types = ['gail'], label = None, limit = None):
    # Read the CSV files into a list of Pandas dataframes
    filenames = []
    labels = []
    for type in types:
        filenames = filenames + [f"{title}-{type}.csv"]
        labels = labels + [f'{type.upper()}']
        if 'gail' in type or 'vail' in type:
            filenames = filenames + [f"{title}-p{type}.csv"]
            labels = labels + [f'protagonist_{type.upper()}']
 
    print(filenames)
    print(labels)
    dfs = [pd.read_csv(filename) for filename in filenames]
    dfs = [df[df['update'] != 'update'] for df in dfs]
    min_steps = limit
    if min_steps is None:
        min_steps = int(min(dfs[0]['update'].astype('int64').max(), dfs[1]['update'].astype('int64').max()))
    print(min_steps)
    # Smooth each dataframe using the defined function
    smooth_dfs = []
    for i, _ in enumerate(dfs):
         
        if 'protagonist' not in labels[i]:
            smooth_dfs.append(
                (dfs[i]["frames"][dfs[i]["update"].astype('int64') <= min_steps].astype('int64'), smooth_curve(dfs[i]['rreturn_mean'][dfs[i]["update"].astype('int64') <= min_steps].astype('float'))), 
                )
        else:
            print(dfs[i]["protagonist_num_frames"])
            smooth_dfs.append((dfs[i]["protagonist_num_frames"][dfs[i]["update"].astype('int64') <= min_steps].astype('int64'), smooth_curve(dfs[i]['protagonist_rreturn_mean'][dfs[i]['update'].astype('int64') <= min_steps].astype('float'))), 
                #(dfs[1]["antagonist_num_frames"][dfs[1]["update"].astype('int64') <= min_steps].astype('int64'), smooth_curve(dfs[1]['antagonist_rreturn_mean'][dfs[1]['update'].astype('int64') <= min_steps].astype('float'))), 
                )

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    
   
    # Plot each smoothed dataframe on the axis
    batch = slice(None, min_steps, 1)
    print(len(smooth_dfs))
    for i, (x, (y_mean, y_low, y_high)) in enumerate(smooth_dfs):
        print(i)
        #x_std = df_std["Step"][df_mean["Step"] <= min_steps]
        #print(x.tolist(), y_mean.tolist())
                
        ax.plot(x.tolist()[:min_steps], y_mean.tolist()[:min_steps], label= labels[i])
         
        ax.fill_between(x.tolist()[:min_steps], y_low.tolist()[:min_steps],  y_high[:min_steps], alpha=0.2)

    return fig, ax


    


titles = [
    "MiniGrid-LavaCrossingS9N1-v0"]#,
    #"MiniGrid-DoorKey-6x6-v0",
    #"MiniGrid-SimpleCrossingS9N1-v0",
    #"MiniGrid-SimpleCrossingS9N2-v0",
    #"MiniGrid-SimpleCrossingS9N3-v0"
    #]

 
for idx, title in enumerate(titles):
    fontsize = 15
    if title == "MiniGrid-DoorKey-6x6-v0":
        limit = 146
        loc = 'upper left'
    elif title == "MiniGrid-SimpleCrossingS9N1-v0":
        limit = 500
        loc = 'best'
        fontsize = 15
    elif title == "MiniGrid-SimpleCrossingS9N2-v0":
        limit = 500
        loc = 'best'
        fontsize = 10
    elif title == "MiniGrid-SimpleCrossingS9N3-v0":
        limit = 1000
        loc = 'best'
        fontsize = 10
    else:
        limit = None
        loc = 'right'
    fig, ax = draw(title, ['gail'], limit = limit)
    #if idx == len(titles) - 1:
        # Set the title and legend for the plot
        #ax.set_title(title)
 
    ax.legend(fontsize = fontsize, loc = loc)
    plt.grid()
    plt.xlabel('frames')
    plt.ylabel('average return per episode')
    # Display the plot
    plt.show()