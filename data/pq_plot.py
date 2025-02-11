import polars as pl
import matplotlib.pyplot as plt
import scienceplots

# Import parquet file
df_true = pl.read_parquet('true.parquet')
df_data = pl.read_parquet("data.parquet")

# Prepare Data to Plot
x_true = df_true['x']
y_true = df_true['y']
x_data = df_data['x']
y_data = df_data['y']

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x_data, y_data, '.', label='Data', color='blue', markersize=2, markeredgewidth=0)
    ax.plot(x_true, y_true, label='True', color='red', linewidth=1)
    ax.legend()
    fig.savefig('data_plot.png', dpi=600, bbox_inches='tight')
