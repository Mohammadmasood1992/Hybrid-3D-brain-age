import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # train() and scatter_plot() call plt.show()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
