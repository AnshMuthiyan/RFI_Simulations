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
import flask
import mpld3

unfiltered_results = xr.open_dataarray('RFI_Simulations/jupyter/BPSK_SK_combined.nc')
def makeDaPlots(unfiltered_results, param="SymbolRate", dims=["SymbolRate", "FC", "FS", "M", "SNR", "DC"], SymRt_thersholds=(None,None), FC_thersholds=(None,None), FS_thersholds=(None,None), M_thersholds=(None,None), SNR_thersholds=(None,None), DC_thersholds=(None,None)):
    #remove all parameters combinations with FP > 0.05
    fp_thereshold = 0.05
    fp = unfiltered_results.sel(Metrics="FP") <= fp_thereshold
    results = unfiltered_results.where(fp.expand_dims(Metrics = unfiltered_results.coords['Metrics']), drop=True)

    #filter other parameters
    results = results.sel(SymbolRate=slice(SymRt_thersholds[0],SymRt_thersholds[1]), FC=slice(FC_thersholds[0],FC_thersholds[1]), FS=slice(FS_thersholds[0],FS_thersholds[1]), M=slice(M_thersholds[0],M_thersholds[1]), SNR=slice(SNR_thersholds[0],SNR_thersholds[1]), DC=slice(DC_thersholds[0],DC_thersholds[1]))
    

    # met = "precision"
    param = "param"
    dims = dims

    TPflattened = results.sel(Metrics="TP").stack(all_dims=dims)
    TPcleaned = TPflattened.dropna("all_dims", how="all")
    TPcleaned = TPcleaned.where(TPcleaned !=0 , drop=True)
    TPcleaned.data = xr.where(np.isfinite(TPcleaned.data), TPcleaned.data, np.nan)

    TPaxis = TPcleaned.indexes["all_dims"].get_level_values(param)
    TPaverages = TPcleaned.groupby(param).mean(dim="all_dims", skipna=True)
    TPSTD = TPcleaned.groupby(param).std(dim="all_dims", skipna=True)
    TPmedians = TPcleaned.groupby(param).median(dim="all_dims", skipna=True)

    print(f"Averages for TP grouped by {param}:")
    print(TPaverages)
    print(f'Shape of the flattened data: {TPflattened.shape}')
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    axs[0, 0].scatter(TPaxis, TPcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
    axs[0, 0].scatter(np.unique(TPaxis), TPaverages, color='red', label=f"Mean TP")
    axs[0, 0].errorbar(np.unique(TPaxis), TPaverages, yerr=TPSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of {met}")
    axs[0, 0].scatter(np.unique(TPaxis), TPmedians, color='blue', label=f"Median TP")
    axs[0, 0].set_ylim(np.min(TPaverages-TPSTD)*1.2, np.max(TPaverages+TPSTD)*1.2)
    axs[0, 0].set_xlabel("Duty Cycle")
    axs[0, 0].set_ylabel("True Positive Rate")
    axs[0, 0].set_title(f"Spectral Kurtosis TP vs {param}")


    FPflattened = results.sel(Metrics="FP").stack(all_dims=dims)
    FPcleaned = FPflattened.dropna("all_dims", how="all")
    FPcleaned = FPcleaned.where(FPcleaned !=0 , drop=True)
    FPcleaned.data = xr.where(np.isfinite(FPcleaned.data), FPcleaned.data, np.nan)

    FPaxis = FPcleaned.indexes["all_dims"].get_level_values(param)
    FPaverages = FPcleaned.groupby(param).mean(dim="all_dims", skipna=True)
    FPSTD = FPcleaned.groupby(param).std(dim="all_dims", skipna=True)
    FPMedians = FPcleaned.groupby(param).median(dim="all_dims", skipna=True)

    axs[0, 1].scatter(FPaxis, FPcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
    axs[0, 1].scatter(np.unique(FPaxis), FPaverages, color='red', label=f"Mean FP")
    axs[0, 1].errorbar(np.unique(FPaxis), FPaverages, yerr=FPSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of FP")
    axs[0, 1].scatter(np.unique(FPaxis), FPMedians, color='blue', label=f"Median FP")
    axs[0, 1].set_ylim(np.min(FPaverages-FPSTD)*1.2, np.max(FPaverages+FPSTD)*1.2)
    axs[0, 1].set_xlabel("Duty Cycle")
    axs[0, 1].set_ylabel("False Positive Rate")
    axs[0, 1].set_title(f"Spectral Kurtosis FP vs {param}")

    PRflattened = results.sel(Metrics="precision").stack(all_dims=dims)
    PRcleaned = PRflattened.dropna("all_dims", how="all")
    PRcleaned = PRcleaned.where(PRcleaned !=0 , drop=True)
    PRcleaned.data = xr.where(np.isfinite(PRcleaned.data), PRcleaned.data, np.nan)

    PRaxis = PRcleaned.indexes["all_dims"].get_level_values(param)
    PRaverages = PRcleaned.groupby(param).mean(dim="all_dims", skipna=True)
    PRSTD = PRcleaned.groupby(param).std(dim="all_dims", skipna=True)
    PRMedians = PRcleaned.groupby(param).median(dim="all_dims", skipna=True)

    axs[1, 0].scatter(PRaxis, PRcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
    axs[1, 0].scatter(np.unique(PRaxis), PRaverages, color='red', label=f"Mean precision")
    axs[1, 0].errorbar(np.unique(PRaxis), PRaverages, yerr=PRSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of precision")
    axs[1, 0].scatter(np.unique(PRaxis), PRMedians, color='blue', label=f"Median precision")
    axs[1, 0].set_ylim(np.min(PRaverages-PRSTD)*1.2, np.max(PRaverages+PRSTD)*1.2)
    axs[1, 0].set_xlabel("Duty Cycle")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].set_title(f"Spectral Kurtosis precision vs {param}")

    ACflattened = results.sel(Metrics="accuracy").stack(all_dims=dims)

    ACcleaned = ACflattened.dropna("all_dims", how="all")
    ACcleaned = ACcleaned.where(ACcleaned !=0 , drop=True)
    ACcleaned.data = xr.where(np.isfinite(ACcleaned.data), ACcleaned.data, np.nan)

    ACaxis = ACcleaned.indexes["all_dims"].get_level_values(param)
    ACaverages = ACcleaned.groupby(param).mean(dim="all_dims", skipna=True)
    ACSTD = ACcleaned.groupby(param).std(dim="all_dims", skipna=True)
    ACMedians = ACcleaned.groupby(param).median(dim="all_dims", skipna=True)

    axs[1, 1].scatter(ACaxis, ACcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
    axs[1, 1].scatter(np.unique(ACaxis), ACaverages, color='red', label=f"Mean accuracy")
    axs[1, 1].errorbar(np.unique(ACaxis), ACaverages, yerr=ACSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of accuracy")
    axs[1, 1].scatter(np.unique(ACaxis), ACMedians, color='blue', label=f"Median accuracy")
    axs[1, 1].set_ylim(np.min(ACaverages-ACSTD)*1.2, np.max(ACaverages+ACSTD)*1.2)
    axs[1, 1].set_xlabel("Duty Cycle")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].set_title(f"Spectral Kurtosis accuracy vs {param}")

    plt.legend()
    html_plot = mpld3.fig_to_html(fig)
    return  html_plot



#remove all parameters combinations with FP > 0.05
fp_thereshold = 0.05
fp = unfiltered_results.sel(Metrics="FP") <= fp_thereshold
results = unfiltered_results.where(fp.expand_dims(Metrics = unfiltered_results.coords['Metrics']), drop=True)

#filter other parameters
SymbolRate_thersholds = (300,0)
results = results.sel(SymbolRate=slice(2,150), FC=slice(None, None), FS=slice(None, None), M=slice(None, None), SNR=slice(None, None), DC=slice(None, None))

met = "precision"
param = "SymbolRate"
dims = ["SymbolRate", "FC", "FS", "M", "SNR", "DC"]

TPflattened = results.sel(Metrics="TP").stack(all_dims=dims)
TPcleaned = TPflattened.dropna("all_dims", how="all")
TPcleaned = TPcleaned.where(TPcleaned !=0 , drop=True)
TPcleaned.data = xr.where(np.isfinite(TPcleaned.data), TPcleaned.data, np.nan)

TPaxis = TPcleaned.indexes["all_dims"].get_level_values(param)
TPaverages = TPcleaned.groupby(param).mean(dim="all_dims", skipna=True)
TPSTD = TPcleaned.groupby(param).std(dim="all_dims", skipna=True)
TPmedians = TPcleaned.groupby(param).median(dim="all_dims", skipna=True)

print(f"Averages for TP grouped by {param}:")
print(TPaverages)
print(f'Shape of the flattened data: {TPflattened.shape}')
fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
axs[0, 0].scatter(TPaxis, TPcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
axs[0, 0].scatter(np.unique(TPaxis), TPaverages, color='red', label=f"Mean TP")
axs[0, 0].errorbar(np.unique(TPaxis), TPaverages, yerr=TPSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of {met}")
axs[0, 0].scatter(np.unique(TPaxis), TPmedians, color='blue', label=f"Median TP")
axs[0, 0].set_ylim(np.min(TPaverages-TPSTD)*1.2, np.max(TPaverages+TPSTD)*1.2)
axs[0, 0].set_xlabel("Duty Cycle")
axs[0, 0].set_ylabel("True Positive Rate")
axs[0, 0].set_title(f"Spectral Kurtosis TP vs {param}")


FPflattened = results.sel(Metrics="FP").stack(all_dims=dims)
FPcleaned = FPflattened.dropna("all_dims", how="all")
FPcleaned = FPcleaned.where(FPcleaned !=0 , drop=True)
FPcleaned.data = xr.where(np.isfinite(FPcleaned.data), FPcleaned.data, np.nan)

FPaxis = FPcleaned.indexes["all_dims"].get_level_values(param)
FPaverages = FPcleaned.groupby(param).mean(dim="all_dims", skipna=True)
FPSTD = FPcleaned.groupby(param).std(dim="all_dims", skipna=True)
FPMedians = FPcleaned.groupby(param).median(dim="all_dims", skipna=True)

axs[0, 1].scatter(FPaxis, FPcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
axs[0, 1].scatter(np.unique(FPaxis), FPaverages, color='red', label=f"Mean FP")
axs[0, 1].errorbar(np.unique(FPaxis), FPaverages, yerr=FPSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of FP")
axs[0, 1].scatter(np.unique(FPaxis), FPMedians, color='blue', label=f"Median FP")
axs[0, 1].set_ylim(np.min(FPaverages-FPSTD)*1.2, np.max(FPaverages+FPSTD)*1.2)
axs[0, 1].set_xlabel("Duty Cycle")
axs[0, 1].set_ylabel("False Positive Rate")
axs[0, 1].set_title(f"Spectral Kurtosis FP vs {param}")

PRflattened = results.sel(Metrics="precision").stack(all_dims=dims)
PRcleaned = PRflattened.dropna("all_dims", how="all")
PRcleaned = PRcleaned.where(PRcleaned !=0 , drop=True)
PRcleaned.data = xr.where(np.isfinite(PRcleaned.data), PRcleaned.data, np.nan)

PRaxis = PRcleaned.indexes["all_dims"].get_level_values(param)
PRaverages = PRcleaned.groupby(param).mean(dim="all_dims", skipna=True)
PRSTD = PRcleaned.groupby(param).std(dim="all_dims", skipna=True)
PRMedians = PRcleaned.groupby(param).median(dim="all_dims", skipna=True)

axs[1, 0].scatter(PRaxis, PRcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
axs[1, 0].scatter(np.unique(PRaxis), PRaverages, color='red', label=f"Mean precision")
axs[1, 0].errorbar(np.unique(PRaxis), PRaverages, yerr=PRSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of precision")
axs[1, 0].scatter(np.unique(PRaxis), PRMedians, color='blue', label=f"Median precision")
axs[1, 0].set_ylim(np.min(PRaverages-PRSTD)*1.2, np.max(PRaverages+PRSTD)*1.2)
axs[1, 0].set_xlabel("Duty Cycle")
axs[1, 0].set_ylabel("Precision")
axs[1, 0].set_title(f"Spectral Kurtosis precision vs {param}")

ACflattened = results.sel(Metrics="accuracy").stack(all_dims=dims)

ACcleaned = ACflattened.dropna("all_dims", how="all")
ACcleaned = ACcleaned.where(ACcleaned !=0 , drop=True)
ACcleaned.data = xr.where(np.isfinite(ACcleaned.data), ACcleaned.data, np.nan)

ACaxis = ACcleaned.indexes["all_dims"].get_level_values(param)
ACaverages = ACcleaned.groupby(param).mean(dim="all_dims", skipna=True)
ACSTD = ACcleaned.groupby(param).std(dim="all_dims", skipna=True)
ACMedians = ACcleaned.groupby(param).median(dim="all_dims", skipna=True)

axs[1, 1].scatter(ACaxis, ACcleaned, alpha=0.1, s=0.7, label=f"{param} Values")
axs[1, 1].scatter(np.unique(ACaxis), ACaverages, color='red', label=f"Mean accuracy")
axs[1, 1].errorbar(np.unique(ACaxis), ACaverages, yerr=ACSTD, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of accuracy")
axs[1, 1].scatter(np.unique(ACaxis), ACMedians, color='blue', label=f"Median accuracy")
axs[1, 1].set_ylim(np.min(ACaverages-ACSTD)*1.2, np.max(ACaverages+ACSTD)*1.2)
axs[1, 1].set_xlabel("Duty Cycle")
axs[1, 1].set_ylabel("Accuracy")
axs[1, 1].set_title(f"Spectral Kurtosis accuracy vs {param}")

plt.legend()
html_plot = mpld3.fig_to_html(fig)
# plt.show()


# plt.close("all")
# CornerDS = results.sel(Metrics="accuracy").stack(all_dims=dims)
# DSTP = results.sel(Metrics="TP").stack(all_dims=dims)
# DSFP = results.sel(Metrics="FP").stack(all_dims=dims)
# DSPR = results.sel(Metrics="precision").stack(all_dims=dims)
# print (f"CornerDS shape: {CornerDS.shape}")
# print (f'CornerDS dim and coords: {CornerDS.dims}, {CornerDS.coords}')
# dfacc= CornerDS.to_dataframe(name="accuracy").dropna()
# dfTP = DSTP.to_dataframe(name="TP").dropna() 
# dfFP = DSFP.to_dataframe(name="FP").dropna()
# dfPR = DSPR.to_dataframe(name="precision").dropna()
# df = dfacc.copy()
# df['TP'] = dfTP['TP']
# df['FP'] = dfFP['FP']
# df['precision'] = dfPR['precision']

# df.to_csv("bpsk_data.csv", index=False)
# df = df.replace([np.inf, -np.inf], np.nan).dropna()

# df = df[['SymbolRate', 'FC', 'FS', 'M', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision']]

# print (f"DataFrame shape: {df.shape}")
# print (f"DataFrame head:\n{df}")

# # 2. Extract the data into an array of shape (samples, dimensions)
# samples = df.values

# # 3. Create the corner plot
# figure = corner.corner(
#     samples, 
#     labels=['SymbolRate', 'FC', 'FS', 'M', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision' ],
#     show_titles=True
# )

# plt.show()

# #FILTERING DATA AND THEN CORNERING. Acc > 1

# # df = df.dropna("all_dims", how="any")
# df = df[df["accuracy"] > 1]

# print(f"Filtered DataFrame shape: {df.shape}")

# samples = df.values

# # 3. Create the corner plot
# figure = corner.corner(
#     samples, 
#     color = 'blue',
#     labels=['SymbolRate', 'FC', 'FS', 'M', 'SNR', 'DC', 'accuracy', 'TP', 'FP', 'precision'],
#     show_titles=True
# )
# plt.savefig("corner_plot_filtered.png")


# plt.show()

# df.to_csv("bpsk_accuracy_SK_accLargerThan1.csv", index=False)

# #plot of SNR vs DC

# param1 = "SNR"
# param2 = "DC"
# P1 = df[param1]
# P2 = df[param2]

# # plt.hexbin(P1, P2)
# plt.scatter(P1, P2, alpha=0.006)
# plt.xlabel(param1)
# plt.ylabel(param2)
# plt.title(f"{param1} vs {param2}")

# plt.savefig(f"{param1}_{param2}_hexbin.png")
# plt.show()


app = flask.Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Metric Plots</title>
</head>
<body>
    <h2>Metric Plots</h2>
    <div>
        {html_plot}
    </div>
    <div>
        <input type="text" id="paramInput" placeholder="SymbolRate_max">
        <input type="text" id="paramInput2" placeholder="SymbolRate_min">
    </div>
    <button onclick="updatePlot()">Update Plot</button>
    <script>
        function updatePlot() {
            const paramValue = document.getElementById('paramInput').value;
            const paramValue2 = document.getElementById('paramInput2').value;

            fetch(`/update_plot?param=${paramValue}&param2=${paramValue2}`)
                .then(response => response.text())
                .then(html => {
                    document.querySelector('div').innerHTML = html;
                })
                .catch(error => console.error('Error updating plot:', error));
        }


    </script>

</body>
</html>
"""

@app.route('/')
def index():
    return html_template.format(html_plot=html_plot)

def update_plot():
    param_value = flask.request.args.get('param', default=None, type=float)
    param_value2 = flask.request.args.get('param2', default=None, type=float)

    # Call the makeDaPlots function with the new parameter values
    html_plot_updated = makeDaPlots(unfiltered_results, SymRt_thersholds=(param_value2, param_value))

    return html_plot_updated

if __name__ == '__main__':
    app.run(debug=True)

