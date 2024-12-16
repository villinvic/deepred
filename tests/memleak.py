import psutil
import time
import signal
import sys
import plotly.graph_objects as go

timestamps = []
ram_usages = []

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Generating plot...")
    plot_ram_usage()
    sys.exit(0)

def plot_ram_usage():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=ram_usages, mode='lines', name='RAM Usage'))
    fig.update_layout(
        title='RAM Usage Over Time',
        xaxis_title='Time (s)',
        yaxis_title='RAM Usage (MB)',
        template='plotly_dark'
    )
    fig.show()

signal.signal(signal.SIGINT, signal_handler)
print("Monitoring RAM usage. Press Ctrl+C to stop and view the plot.")

start_time = time.time()

try:
    while True:
        current_time = time.time() - start_time
        ram_info = psutil.virtual_memory()
        ram_usage_mb = ram_info.used / (1024 * 1024)  # Convert bytes to MB

        timestamps.append(current_time)
        ram_usages.append(ram_usage_mb)
        time.sleep(1)

except KeyboardInterrupt:
    signal_handler(None, None)
