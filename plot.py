import os, re
import torch
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import numpy as np

# Directories
LOGS_DIR = "logs"
PLOTS_DIR = "plots"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper function to check if a value is plottable (i.e., list of numbers)
def is_plottable(value):
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)
def plot(x, y, x_label, y_label, title, filename, x_scale, y_scale , legend = None,xlim = [0, 1e5], ylim = [0,100], color = None, linestyle = None):
    # check if x and y are lists
    if not isinstance(x, list) or not isinstance(y, list):
        raise ValueError("x and y must be lists")
    plt.figure(figsize=(10, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], color = color[i], linestyle = linestyle[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        if legend is not None:
            plt.legend(legend)
        plt.xscale(x_scale)
        plt.yscale(y_scale)
    plt.ylim(ylim)
    plt.xlim(xlim)
    # Clean filename
    plot_filename = f"{filename[:-3]}_{x_label}_vs_{y_label}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi = 300)
    plt.close()
    
lower_costs = []
lower_costs_epoch = []
avg_psnr = []
avg_loss = []
loss = []
psnr_batch = []
avg_loss_test = []
avg_psnr_test = []
cg_iter = []
lower_iter = []
legend = []
epochs = []
color = []
linestyle = []
HIGH_VALUE = 1e6

r_avg_loss = []
r_avg_psnr = []
r_avg_lower_costs = []

warm_colors = [
    'red', 'orange', 'darkred', 'salmon', 'coral', 'sienna', 'tomato', 'chocolate',
    'firebrick', 'gold', 'orangered', 'darkorange', 'lightcoral', 'peru', 'crimson'
]
cool_colors = [
    'blue', 'cyan', 'purple', 'teal', 'turquoise', 'skyblue', 'indigo',
    'dodgerblue', 'deepskyblue', 'navy', 'mediumblue', 'steelblue', 'darkblue', 'slateblue', 'royalblue'
]
mixed_colors = [
    'red', 'blue', 'orange', 'cyan', 'darkred', 'purple', 'salmon', 'teal',
    'coral', 'turquoise', 'sienna', 'skyblue', 'tomato', 'indigo', 'chocolate',
    'dodgerblue', 'firebrick', 'deepskyblue', 'gold', 'navy', 'orangered',
    'mediumblue', 'darkorange', 'steelblue', 'lightcoral', 'darkblue', 'peru',
    'slateblue', 'crimson', 'royalblue'
]

linestyles = ['-', '--', '-.', ':']
# based in alpha
# def extract_first_number(filename):
#     match = re.search(r"logs_([0-9.eE+-]+)", filename)
#     if match:
#         return float(match.group(1))
#     return float('inf')  # Push malformed files to the end

# based on eps
def extract_two_numbers(filename):
    base = os.path.splitext(filename)[0]  # remove .pt or any extension
    match = re.search(r"logs_([\d._eE+-]+)", base)
    if match:
        try:
            float_vals = [float(val) for val in match.group(1).split("_") if val != ""]
            # Sort primarily by second value, then by first
            second = float_vals[1] if len(float_vals) > 1 else float('inf')
            first = float_vals[0] if len(float_vals) > 0 else float('inf')
            return (second, first)
        except ValueError:
            pass
    return (float('inf'), float('inf'))  # fallback if format is off
# Sort by the first float after "logs_"
sorted_files = sorted(
    [f for f in os.listdir(LOGS_DIR) if f.startswith("logs_")],
    key=extract_two_numbers
)
# Loop through each .pt log file
for filename in sorted_files:
    if filename.endswith(".pt"):
        log_path = os.path.join(LOGS_DIR, filename)
        log_data = torch.load(log_path, map_location='cpu', weights_only = False, pickle_module=pickle)

        # Extract plottable entries
        numeric_entries = {k: v for k, v in log_data.items() if is_plottable(v) and len(v) > 1}
        lower_costs_epoch.append(log_data["cost_epoch"])
        avg_psnr.append([-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_epoch"]])
        avg_loss.append([HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_epoch"]])
        loss.append([log_data["loss_batch"][0]] + [HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_batch"]])
        psnr_batch.append([log_data["psnr_batch"][0]] + [-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_batch"]])
        lower_costs.append([0] + log_data["cost_batch"])
        avg_loss_test.append([HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_test"]])
        avg_psnr_test.append([-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_test"]])
        cg_iter.append(log_data["cg_batch"])
        lower_iter.append(log_data["lower_batch"])

        # Compute running averages using convolution
        window_size = 5 # Define the window size for the running average
        r_avg_loss.append(np.convolve(loss[-1], np.ones(window_size) / window_size, mode='valid'))
        r_avg_lower_costs.append(np.convolve(lower_costs[-1], np.ones(window_size) / window_size, mode='valid'))
        r_avg_psnr.append(np.convolve(psnr_batch[-1], np.ones(window_size) / window_size, mode='valid'))
        # Extract parameters
        print(f"{log_data['step']} {log_data['q']} {log_data['p']}")
        if log_data['q'] ==0 and log_data['p'] ==0:
            legend.append(
            rf"$\epsilon$: ${float(log_data['eps']):.0e}$ "
            rf"$\alpha$: ${float(log_data['step']):.0e}$ "
            )
        else:
            legend.append(
                rf"$\epsilon$: ${float(log_data['eps']):.0e}$ "
                rf"$\alpha$: ${float(log_data['step']):.0e}$ "
                rf"$q$: ${float(log_data['q']):.0e}$ "
                rf"$p$: ${float(log_data['p']):.1e}$"
            )
        epochs.append(np.arange(len(log_data["psnr_test"])))
        
        if log_data['p'] == 0 and log_data['q'] == 0:
            linestyle.append(linestyles[0])
        elif log_data['p'] != 0 and log_data['q'] == 0:
            linestyle.append(linestyles[1])
        elif log_data['p'] == 0 and log_data['q'] != 0:
            linestyle.append(linestyles[2])
        else:
            linestyle.append(linestyles[0])
        
        if log_data['p'] == 0 or log_data['q'] == 0:
            color.append(cool_colors[0])
            cool_colors.pop(0)
        else:
            color.append(mixed_colors[0])
            mixed_colors.pop(0)


plot(lower_costs_epoch, avg_psnr, "Total lower-level computations", "PSNR", "Total Lower Costs vs Average PSNR per Epoch", "lower_psnr.png", "linear", "linear", legend,xlim=[0,1e4], ylim=[15,30], color= color,linestyle = linestyle)
plot(lower_costs_epoch, avg_loss, "Total lower-level computations", "Loss", "Total Lower Costs vs Average Loss per Epoch", "lower_loss.png", "symlog", "log", legend,xlim=[0,1e4], ylim=[2,120], color= color,linestyle = linestyle)
plot(lower_costs, loss, "Total lower-level cost", "Loss", "Total Lower Costs vs Loss per batch", "lower_loss_iter.png", "symlog", "log", legend,xlim=[0,1e4], ylim=[2,120], color= color,linestyle = linestyle)
plot(epochs, avg_loss_test, "Epoch", "Test Loss", "Epoch vs Average Test Loss per Epoch", "lower_loss_test.png", "linear", "linear", legend,xlim=[0,120], ylim=[50,1000], color= color,linestyle = linestyle)
plot(epochs, avg_psnr_test, "Epoch", "Test PSNR", "Epoch vs Average Test PSNR per Epoch", "lower_psnr_test.png", "linear", "linear", legend,xlim=[0,120], ylim=[15,30], color= color,linestyle = linestyle)

plot(r_avg_lower_costs, r_avg_psnr, "Total lower-level computations", "PSNR", "Total Lower Costs vs Average PSNR per Epoch", "lower_psnr_mv_average.png", "linear", "linear", legend,xlim=[0,1e4], ylim=[15,30], color= color,linestyle = linestyle)
plot(r_avg_lower_costs, r_avg_loss, "Total lower-level computations", "Loss", "Total Lower Costs vs Average Loss per Epoch", "lower_loss_mv_average.png", "symlog", "log", legend,xlim=[0,1e4], ylim=[2,120], color= color,linestyle = linestyle)

print(f"Plots saved to '{PLOTS_DIR}/'")