import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy as sp
import scipy.stats as stats
import warnings as warn
import pandas as pd
import corner 
import xarray as xr
import torch

RFIMit = "msSK"
SigGen = "QPSK" 
dims = ["SymbolRate", "FC", "FS", "M", "n", "SNR", "DC"]

unfiltered_results = xr.open_dataarray('RFI_Simulations/jupyter/QPSK_msSK_full.nc')

default_FP_thers = 0.05
metrics = [
    ("TP", "True Positive Rate"),
    ("FP", "False Positive Rate"),
    ("precision", "Precision"),
    ("accuracy", "Accuracy"),
    ("time", "Execution Time (s)"),
]

param = "M"
color_param = "n"
thresholds = None
fp_threshold = default_FP_thers
# def makeDaPlots(unfiltered_results, param="SymbolRate", dims=dims, thresholds=None, fp_threshold=default_FP_thers):
if param not in dims:
    raise ValueError(f"'{param}' is not one of the available parameters: {dims}")
if color_param is not None and color_param not in dims:
    raise ValueError(f"'{color_param}' is not one of the available parameters: {dims}")
if color_param == param:
    color_param = None

thresholds = thresholds or {}

#make mask for FP thresholding
fp = unfiltered_results.sel(Metrics="FP") <= fp_threshold
#print percent of data points that pass the FP threshold
percent_pass = fp.sum().item() / fp.size * 100
print(f"Percent of data points that pass the FP threshold of {fp_threshold}: {percent_pass:.2f}%")
#apply that mask
results = unfiltered_results.where(fp.expand_dims(Metrics=unfiltered_results.coords['Metrics']), drop=True)

sel_kwargs = {d: slice(*thresholds.get(d, (None, None))) for d in dims}
results = results.sel(**sel_kwargs)

fig, axs = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)

for ax, (metric, ylabel) in zip(axs.flatten(), metrics):
    flattened = results.sel(Metrics=metric).stack(all_dims=dims)
    cleaned = flattened.dropna("all_dims", how="all")
    cleaned = cleaned.where(cleaned != 0, drop=True)
    cleaned.data = xr.where(np.isfinite(cleaned.data), cleaned.data, np.nan)

    axis_vals = cleaned.indexes["all_dims"].get_level_values(param)
    color_vals = cleaned.indexes["all_dims"].get_level_values(color_param) if color_param else None
    averages = cleaned.groupby(param).mean(dim="all_dims", skipna=True)
    std = cleaned.groupby(param).std(dim="all_dims", skipna=True)
    medians = cleaned.groupby(param).median(dim="all_dims", skipna=True)
    colmed = cleaned.groupby(color_param).median(dim="all_dims", skipna=True) if color_param else None

    print(f"Averages for {metric} grouped by {param}:")
    print(averages)
    print(f"Shape of the flattened data: {flattened.shape}")

    if color_param is not None:
        color_vals = cleaned.indexes["all_dims"].get_level_values(color_param)
        scatter = ax.scatter(axis_vals, cleaned, c=color_vals, cmap='magma', alpha=0.4, s=1.2, label=f"{param} Values")
        ax.scatter(np.unique(color_vals), colmed, color='green', label=f"Median {color_param}")  
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_param)
    else:
        ax.scatter(axis_vals, cleaned, alpha=0.2, s=1.2, label=f"{param} Values")
    ax.scatter(np.unique(axis_vals), averages, color='red', label=f"Mean {metric}")
    ax.errorbar(np.unique(axis_vals), averages, yerr=std, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of {metric}")
    ax.scatter(np.unique(axis_vals), medians, color='blue', label=f"Median {metric}")
    ax.set_ylim(np.nanmin(averages - std) * 1.2, np.nanmax(averages + std) * 1.2)
    ax.set_xlabel(param)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{RFIMit} and {SigGen} {metric} vs {param}")
    ax.legend(loc='upper right')

plt.show()


# plt.close("all")
CornerDS = results.sel(Metrics="accuracy").stack(all_dims=["SymbolRate", "FC", "FS", "M", "n", "SNR", "DC"])
DSTP = results.sel(Metrics="TP").stack(all_dims=["SymbolRate", "FC", "FS", "M", "n", "SNR", "DC"])
DSFP = results.sel(Metrics="FP").stack(all_dims=["SymbolRate", "FC", "FS", "M", "n", "SNR", "DC"])
DSPR = results.sel(Metrics="precision").stack(all_dims=["SymbolRate", "FC", "FS", "M", "n", "SNR", "DC"])
print (f"CornerDS shape: {CornerDS.shape}")
print (f'CornerDS dim and coords: {CornerDS.dims}, {CornerDS.coords}')
dfacc= CornerDS.to_dataframe(name="accuracy").dropna()
dfTP = DSTP.to_dataframe(name="TP").dropna() 
dfFP = DSFP.to_dataframe(name="FP").dropna()
dfPR = DSPR.to_dataframe(name="precision").dropna()
df = dfacc.copy()
df['TP'] = dfTP['TP']
df['FP'] = dfFP['FP']
df['precision'] = dfPR['precision']

df.to_csv("bpsk_data.csv", index=False)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

df = df[['SymbolRate', 'FC', 'FS', 'M', 'n', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision']]

print (f"DataFrame shape: {df.shape}")
print (f"DataFrame head:\n{df}")

# 2. Extract the data into an array of shape (samples, dimensions)
samples = df.values

# 3. Create the corner plot
figure = corner.corner(
    samples, 
    color = 'blue',
    labels=['SymbolRate', 'FC', 'FS', 'Count', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision' ],
    show_titles=True
)

plt.show()

#FILTERING DATA AND THEN CORNERING. Acc > 1

# df = df.dropna("all_dims", how="any")
df = df[df["accuracy"] > 1]

print(f"Filtered DataFrame shape: {df.shape}")

samples = df.values

# 3. Create the corner plot
figure = corner.corner(
    samples, 
    color = 'blue',
    labels=['SymbolRate', 'FC', 'FS', 'Count', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision'],
    show_titles=True
)
plt.savefig("corner_plot_filtered.png")


plt.show()

df.to_csv("bpsk_accuracy_SK_accLargerThan1.csv", index=False)

#plot of SNR vs DC

param1 = "SNR"
param2 = "DC"
P1 = df[param1]
P2 = df[param2]

# plt.hexbin(P1, P2)
plt.scatter(P1, P2, alpha=0.006)
plt.xlabel(param1)
plt.ylabel(param2)
plt.title(f"{param1} vs {param2}")

plt.savefig(f"{param1}_{param2}_hexbin.png")
plt.show()