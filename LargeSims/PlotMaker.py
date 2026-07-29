import json
import traceback
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
#For SK
# RFIMit = "SK"
# SigGen = "BPSK"
# dims = ["SymbolRate", "FC", "FS", "M", "SNR", "DC"]

#For msSK
# RFIMit = "msSK"
# SigGen = "BPSK"
# dims = ["SymbolRate", "FC", "FS", "M", 'n', "SNR", "DC"]

# #For ConvRFI
# RFIMit = "ConvRFI"
# SigGen = "BPSK"
# dims = ["SymbolRate", "FC", "FS", "AggressionFactor1", "AggressionFactor2", "AggressionFactor3", "AggressionFactor4", "Bins", "SNR", "DC"]

# #For AOFlagger
RFIMit = "AOFlagger"
SigGen = "BPSK" 
dims = ["SymbolRate", "FC", "FS", "Count", "SNR", "DC"]

unfiltered_results = xr.open_dataarray('RFI_Simulations/jupyter/BPSK_AOFlagger_combined.nc')

default_FP_thers = 0.05
metrics = [
    ("TP", "True Positive Rate"),
    ("FP", "False Positive Rate"),
    ("precision", "Precision"),
    ("accuracy", "Accuracy"),
    ("time", "Execution Time (s)"),
]



def makeDaPlots(unfiltered_results, param="SymbolRate", dims=dims, thresholds=None, fp_threshold=default_FP_thers):
    if param not in dims:
        raise ValueError(f"'{param}' is not one of the available parameters: {dims}")

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
        averages = cleaned.groupby(param).mean(dim="all_dims", skipna=True)
        std = cleaned.groupby(param).std(dim="all_dims", skipna=True)
        medians = cleaned.groupby(param).median(dim="all_dims", skipna=True)

        print(f"Averages for {metric} grouped by {param}:")
        print(averages)
        print(f"Shape of the flattened data: {flattened.shape}")

        ax.scatter(axis_vals, cleaned, alpha=0.2, s=1.2, label=f"{param} Values")
        ax.scatter(np.unique(axis_vals), averages, color='red', label=f"Mean {metric}")
        ax.errorbar(np.unique(axis_vals), averages, yerr=std, fmt='o', color='red', ecolor='gray', elinewidth=.6, capsize=3, label=f"STD of {metric}")
        ax.scatter(np.unique(axis_vals), medians, color='blue', label=f"Median {metric}")
        ax.set_ylim(np.nanmin(averages - std) * 1.2, np.nanmax(averages + std) * 1.2)
        ax.set_xlabel(param)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{RFIMit} and {SigGen} {metric} vs {param}")

    plt.legend()
    html_plot = mpld3.fig_to_html(fig)
    plt.close(fig)
    return html_plot


PARAM_INPUTS_HTML = "".join(
    f'''
        <div class="param-row">
            <span class="param-label">{d}</span>
            <input type="text" id="{d}_min" placeholder="min">
            <input type="text" id="{d}_max" placeholder="max">
        </div>'''
    for d in dims
)

PARAM_OPTIONS_HTML = "".join(
    f'<option value="{d}"{" selected" if d == "SymbolRate" else ""}>{d}</option>'
    for d in dims
)

try:
    html_plot = makeDaPlots(unfiltered_results, param="SymbolRate", dims=dims, thresholds={"SymbolRate": (2, 150)}, fp_threshold=default_FP_thers)
except Exception:
    traceback.print_exc()
    html_plot = "<div class='error'>Failed to generate the initial plot. Check the server console for details.</div>"

app = flask.Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Metric Plots</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        #filter-form { display: flex; flex-wrap: wrap; gap: 1em; align-items: flex-end; margin: 1em 0; }
        .param-row { display: flex; flex-direction: column; }
        .param-row input { width: 6em; }
        #error-box { color: #b00020; white-space: pre-wrap; margin-top: 1em; }
    </style>
</head>
<body>
    <h2>Metric Plots</h2>
    <div id="plot-container">
        __HTML_PLOT__
    </div>

    <div id="filter-form">
        __PARAM_INPUTS__
        <div class="param-row">
            <span class="param-label">X-axis parameter</span>
            <select id="xparam">__PARAM_OPTIONS__</select>
        </div>
        <div class="param-row">
            <span class="param-label">FP threshold</span>
            <input type="text" id="fp_threshold" value="__DEFAULT_FP_THRESHOLD__">
        </div>
    </div>
    <button onclick="updatePlot()">Update Plot</button>
    <div id="error-box"></div>

    <script>
        const dims = __DIMS_JSON__;

        function updatePlot() {
            const params = new URLSearchParams();
            dims.forEach(d => {
                const minVal = document.getElementById(d + '_min').value;
                const maxVal = document.getElementById(d + '_max').value;
                if (minVal) params.append(d + '_min', minVal);
                if (maxVal) params.append(d + '_max', maxVal);
            });
            params.append('xparam', document.getElementById('xparam').value);
            const fpThreshold = document.getElementById('fp_threshold').value;
            if (fpThreshold) params.append('fp_threshold', fpThreshold);

            const errorBox = document.getElementById('error-box');
            errorBox.textContent = '';

            fetch(`/update_plot?${params.toString()}`)
                .then(response => response.text().then(text => ({ ok: response.ok, text })))
                .then(({ ok, text }) => {
                    if (!ok) {
                        errorBox.textContent = text;
                        return;
                    }
                    const container = document.getElementById('plot-container');
                    container.innerHTML = text;
                    container.querySelectorAll('script').forEach(oldScript => {
                        const newScript = document.createElement('script');
                        newScript.textContent = oldScript.textContent;
                        oldScript.replaceWith(newScript);
                    });
                })
                .catch(error => {
                    errorBox.textContent = 'Request failed: ' + error;
                    console.error('Error updating plot:', error);
                });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return (
        html_template
        .replace("__HTML_PLOT__", html_plot)
        .replace("__PARAM_INPUTS__", PARAM_INPUTS_HTML)
        .replace("__PARAM_OPTIONS__", PARAM_OPTIONS_HTML)
        .replace("__DEFAULT_FP_THRESHOLD__", str(default_FP_thers))
        .replace("__DIMS_JSON__", json.dumps(dims))
    )


@app.route('/update_plot')
def update_plot():
    try:
        thresholds = {
            d: (
                flask.request.args.get(f'{d}_min', default=None, type=float),
                flask.request.args.get(f'{d}_max', default=None, type=float),
            )
            for d in dims
        }
        xparam = flask.request.args.get('xparam', default='SymbolRate', type=str)
        fp_threshold = flask.request.args.get('fp_threshold', default=default_FP_thers, type=float)

        return makeDaPlots(unfiltered_results, param=xparam, dims=dims, thresholds=thresholds, fp_threshold=fp_threshold)
    except Exception as e:
        traceback.print_exc()
        return f"Error updating plot: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
