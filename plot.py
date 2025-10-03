import os, re
import torch
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import numpy as np
import seaborn as sns
import math

# Directories
LOGS_DIR = "logs/longer_run/1e0"
PLOTS_DIR = "plots/longer_run/1e0"

plt.rcParams["lines.linewidth"] = 2

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper function to check if a value is plottable (i.e., list of numbers)
def is_plottable(value):
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)
def plot(x, y, x_label, y_label, title, filename, x_scale, y_scale , legend = None,xlim = None, ylim = None, color = None, linestyle = None):
    # check if x and y are lists
    if not isinstance(x, list) or not isinstance(y, list):
        raise ValueError("x and y must be lists")
    plt.figure(figsize=(8, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], color = color[i], linestyle = linestyle[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        if legend is not None:
            plt.legend(legend, loc = "best")
        plt.xscale(x_scale)
        plt.yscale(y_scale)
    if ylim is not None:
        if min(ylim) == 0:
            plt.ylim(top = max(ylim))
        elif max(ylim) == np.inf:
            
            plt.ylim(bottom = min(ylim), top = max(y[i])+5)
        else:
            plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    # Clean filename
    plot_filename = f"{filename[:-3]}_{x_label}_vs_{y_label}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
lower_costs = []
lower_costs_epoch = []
upper_iter = []
avg_psnr = []
avg_loss = []
loss = []
psnr_batch = []
avg_loss_test = []
avg_psnr_test = []
cg_iter = []
lower_iter = []
eps = []
lr = []
iterations = []
legend = []
epochs = []
color = []
linestyle = []
HIGH_VALUE = 1e6

r_avg_loss = []
r_avg_psnr = []
r_avg_lower_costs = []
norm_grad_min = []
norm_grad = []
x_norm_grad = []

# warm_colors = [
#     'red', 'orange', 'darkred', 'salmon', 'coral', 'sienna', 'tomato', 'chocolate',
#     'firebrick', 'gold', 'orangered', 'darkorange', 'lightcoral', 'peru', 'crimson'
# ]
# cool_colors = [
#     'blue', 'cyan', 'purple', 'teal', 'turquoise', 'skyblue', 'indigo',
#     'dodgerblue', 'deepskyblue', 'navy', 'mediumblue', 'steelblue', 'darkblue', 'slateblue', 'royalblue'
# ]
# mixed_colors = [
#     'red', 'blue', 'orange', 'cyan', 'darkred', 'purple', 'salmon', 'teal',
#     'coral', 'turquoise', 'sienna', 'skyblue', 'tomato', 'indigo', 'chocolate',
#     'dodgerblue', 'firebrick', 'deepskyblue', 'gold', 'navy', 'orangered',
#     'mediumblue', 'darkorange', 'steelblue', 'lightcoral', 'darkblue', 'peru',
#     'slateblue', 'crimson', 'royalblue'
# ]

cool_colors = sns.color_palette("Set1")
warm_colors = sns.color_palette("Set2")
# mixed_colors = sns.color_palette("husl", n_colors=10)
mixed_colors = [cool_colors[i % len(cool_colors)] for i in range(len(warm_colors))] + \
               [warm_colors[i % len(warm_colors)] for i in range(len(cool_colors))]
linestyles = ['-', '--', ':', '-.']
def format_10_neg_exp(x):
    """Format a float as LaTeX 10^{-x}"""
    exp = int(round(-math.log10(x)))
    return rf"$10^{{-{exp}}}$"
# based in alpha
# def extract_first_number(filename):
#     match = re.search(r"logs_([0-9.eE+-]+)", filename)
#     if match:
#         return float(match.group(1))
#     return float('inf')  # Push malformed files to the end

# based on eps
# def extract_two_numbers(filename):
#     base = os.path.splitext(filename)[0]  # remove .pt or any extension
#     match = re.search(r"logs_([\d._eE+-]+)", base)
#     if match:
#         try:
#             float_vals = [float(val) for val in match.group(1).split("_") if val != ""]
#             # Sort primarily by second value, then by first
#             second = float_vals[1] if len(float_vals) > 1 else float('inf')
#             first = float_vals[0] if len(float_vals) > 0 else float('inf')
#             return (second, first)
#         except ValueError:
#             pass
#     return (float('inf'), float('inf'))  # fallback if format is off

def extract_sort_key(filename):
    base = os.path.splitext(filename)[0]  # remove .pt or any extension
    match = re.search(r"logs_([\d._eE+-]+)", base)
    if match:
        try:
            float_vals = [float(val) for val in match.group(1).split("_") if val != ""]
            # Ensure we have enough values
            alpha = float_vals[0] if len(float_vals) > 0 else float('inf')
            eps = float_vals[1] if len(float_vals) > 1 else float('inf')
            q = float_vals[3] if len(float_vals) > 3 else float('inf')
            p = float_vals[2] if len(float_vals) > 2 else float('inf')
            return (eps, alpha, p, q)
        except ValueError:
            pass
    return (float('inf'), float('inf'), float('inf'), float('inf'))
# Sort by the first float after "logs_"
# sorted_files = sorted(
#     [f for f in os.listdir(LOGS_DIR) if f.startswith("logs_")],
#     key=extract_two_numbers
# )
sorted_files = sorted(
    [f for f in os.listdir(LOGS_DIR) if f.startswith("logs_")],
    key=extract_sort_key
)
# Loop through each .pt log file
for filename in sorted_files:
    if filename.endswith(".pt"):
        log_path = os.path.join(LOGS_DIR, filename)
        log_data = torch.load(log_path, map_location='cpu', pickle_module=pickle)

        # Extract plottable entries
        numeric_entries = {k: v for k, v in log_data.items() if is_plottable(v) and len(v) > 1}
        lower_costs_epoch.append(log_data["cost_epoch"])
        avg_psnr.append([-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_epoch"]])
        avg_loss.append([HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_epoch"]])
        loss.append([log_data["loss_epoch"][0]] + [HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_batch"]])
        psnr_batch.append([log_data["psnr_epoch"][0]] + [-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_batch"]])
        lower_costs.append([0] + log_data["cost_batch"])
        avg_loss_test.append([HIGH_VALUE if not np.isfinite(l) else l for l in log_data["loss_test"]])
        avg_psnr_test.append([-HIGH_VALUE if not np.isfinite(l) else l for l in log_data["psnr_test"]])
        cg_iter.append(log_data["cg_batch"])
        lower_iter.append(log_data["lower_batch"])
        eps.append(log_data["eps"])
        lr.append(log_data["step"])
        running_min_grad_norm = [log_data["grad_norm"][0]]

        for i in range(1, len(log_data["grad_norm"])):
            running_min_grad_norm.append(min(running_min_grad_norm[-1], log_data["grad_norm"][i]))

        norm_grad_min.append(np.array(running_min_grad_norm))
        norm_grad.append(log_data["grad_norm"])
        x_norm_grad.append(np.arange(1, len(log_data["grad_norm"])+1))
        iterations.append(np.arange(len(log_data["eps"])))
        upper_iter.append(np.arange(len(loss[-1])))

        # Compute running averages using convolution
        window_size = 8 # Define the window size for the running average
        # Compute running average and prepend loss_epoch[0] to the result
        r_avg = np.convolve(loss[-1], np.ones(window_size) / window_size, mode='valid')
        r_avg_loss.append(np.concatenate([[np.array(log_data["loss_epoch"][0])], r_avg]))
        r_avg_cost = np.convolve(lower_costs[-1], np.ones(window_size) / window_size, mode='valid')
        r_avg_lower_costs.append(np.concatenate([np.array([0]), r_avg_cost]))
        r_avg_psnr_val = np.convolve(psnr_batch[-1], np.ones(window_size) / window_size, mode='valid')
        r_avg_psnr.append(np.concatenate([np.array([log_data["psnr_epoch"][0]]), r_avg_psnr_val]))
        # Extract parameters
        def format_decimal_auto(x):
            """Format decimal with .1f if 1 decimal digit, else .2f."""
            if round(x, 2) == round(x, 1):
                return f"{x:.1f}"
            else:
                return f"{x:.2f}"
        print(f" {log_data['q']} {log_data['p']}")
        p_val = float(log_data['p'])
        q_val = float(log_data['q'])
        eps_val = float(log_data['eps'][0])
        alpha_val = float(log_data['step'][0])

        legend.append(
            rf"$\epsilon$: {format_10_neg_exp(eps_val)}, "
            rf"$\alpha$: {format_10_neg_exp(alpha_val)}, "
            rf"$p$: ${format_decimal_auto(p_val)}$, "
            rf"$q$: ${format_decimal_auto(q_val)}$"
        )
        epochs.append(np.arange(len(log_data["psnr_test"])))
        
        if log_data['p'] == 0 and log_data['q'] == 0:
            linestyle.append(linestyles[0])
        elif log_data['p'] != 0 and log_data['q'] == 0:
            linestyle.append(linestyles[1])
        elif log_data['p'] == 0 and log_data['q'] != 0:
            linestyle.append(linestyles[2])
        else:
            linestyle.append(linestyles[3])
        
        if log_data['p'] == 0 or log_data['q'] == 0:
            color.append(mixed_colors[0])
            mixed_colors.pop(0)
        elif log_data['p'] != 0 and log_data['q'] != 0:
            color.append(mixed_colors[0])
            mixed_colors.pop(0)
        else:
            color.append(mixed_colors[0])
            mixed_colors.pop(0)


plot(lower_costs_epoch, avg_psnr, "Total lower-level computations", "PSNR", "Total lower costs vs average training PSNR", "lower_psnr.png", "linear", "linear", legend, ylim=[20,30], color= color,linestyle = linestyle)
plot(lower_costs_epoch, avg_loss, "Total lower-level computations", "Loss", "Total lower costs vs average training loss", "lower_loss.png", "symlog", "log", legend, ylim=[10,130], color= color,linestyle = linestyle)
plot(lower_costs, loss, "Total lower-level cost", "Loss", "Total lower-level costs vs trainin loss per batch", "lower_loss_iter.png", "symlog", "log", legend, color= color,linestyle = linestyle , ylim=[10,130])
plot(epochs, avg_loss_test, "Epoch", "Loss", "Epoch vs average test loss per epoch", "lower_loss_test.png", "linear", "log", legend, color= color,linestyle = linestyle, ylim=[100,2000])
plot(epochs, avg_psnr_test, "Epoch", "PSNR", "Epoch vs average test PSNR per epoch", "lower_psnr_test.png", "linear", "linear", legend, ylim=[15,30], color= color,linestyle = linestyle)
plot(lower_costs_epoch, avg_psnr_test, "Total lower-level computations", "PSNR", "Total lower costs vs average test PSNR", "lower_psnr_test_epoch.png", "linear", "linear", legend, ylim=[20,30], color= color,linestyle = linestyle)
plot(r_avg_lower_costs, r_avg_psnr, "Total lower-level computations", "PSNR", "Total lower-level costs vs running average training PSNR", "lower_psnr_mv_average.png", "linear", "linear", legend, ylim=[20,30], color= color,linestyle = linestyle)
plot(r_avg_lower_costs, r_avg_loss, "Total lower-level computations", "Loss", "Total lower-level costs vs running average training loss", "lower_loss_mv_average.png", "symlog", "log", legend, color= color,linestyle = linestyle, ylim=[10,130])
plot(iterations, eps, "Epoch", "Epsilon", "Epoch vs accuracy", "eps_epoch.png", "linear", "log", legend,color= color,linestyle = linestyle)
plot(iterations, lr, "Epoch", "Learning Rate", "Epoch vs upper-level step size", "lr_epoch.png", "linear", "log", legend, color= color,linestyle = linestyle)
plot(upper_iter, loss, "Upper-level iterations", "Loss", "Upper-level iterations vs loss", "upper_loss_iter.png", "symlog", "log", legend, color= color,linestyle = linestyle, ylim=[10,130])
plot(upper_iter, psnr_batch, "Upper-level iterations", "PSNR", "Upper-level iterations vs PSNR", "upper_psnr_iter.png", "linear", "linear", legend, color= color,linestyle = linestyle, ylim=[20,30])
plot(x_norm_grad, norm_grad_min, "Upper-level iterations", "Minimum seen gradient norm", "Upper-level iterations vs gradient norm", "upper_grad_iter.png", "log", "log", legend, color= color,linestyle = linestyle, ylim=[1e-0,1000])
plot(x_norm_grad, norm_grad, "Upper-level iterations", "Gradient norm", "Upper-level iterations vs gradient norm", "upper_grad_iter_all.png", "log", "log", legend, color= color,linestyle = linestyle, ylim=[1e-0,1000])
print(f"Plots saved to '{PLOTS_DIR}/'")