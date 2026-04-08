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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json

# Sample data for fruit sales
fruits = ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry']
sales = np.array([200, 300, 150, 220, 180])

# Create a dictionary to hold the data for the pie chart
manifest = {
    'type': 'pie_chart',
    'labels': fruits,
    'values': sales.tolist(),
    'title': 'Fruit Sales'
}

# Print the manifest in JSON format
print(json.dumps(manifest))

# Create the pie chart
plt.pie(sales, labels=fruits, autopct='%1.1f%%')
plt.title(manifest['title'])
plt.show()