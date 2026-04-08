import numpy as np
import json
import sys
import scipy.stats as stats
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)
sys.stdout.reconfigure(encoding='utf-8')
original_dumps = json.dumps
json.dumps = lambda obj, **kwargs: original_dumps(obj, cls=NpEncoder, **kwargs)
import numpy as np
from scipy.stats import norm
import json

# Define the parameters of the normal distribution
mu = 0
sigma = 1
low = -3
high = 3
num_samples = 100

# Generate the values of the x-axis
x = np.linspace(low, high, num_samples)

# Calculate the values of the normal distribution
y = norm.pdf(x, loc=mu, scale=sigma)

# Create a dictionary to represent the manifest
manifest = {
    "x": x.tolist(),
    "y": y.tolist()
}

# Print the manifest as a JSON object
print(json.dumps(manifest))