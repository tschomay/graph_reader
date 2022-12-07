###########
# History #
###########
'''
V2:
- Added smoothing function
- Updated constant names
V1:
- Initial version
'''

###########
# Imports #
###########
import pandas as pd
import numpy as np
from imageio import imread

#############
# Constants #
#############
INPUT_GRAPH_FILENAME = 'graph.png'
OUTPUT_CSV_FILENAME = 'graph_values.csv'
WHITE_THRESHOLD = 100  # Greyscale values above this are considered white
WIN_SIZE = 5  # Size of window for smoothing
# The following values set graph scale
X_MIN = 0
X_MAX = 150
Y_MIN = 0
Y_MAX = 150

#############
# Main Code #
#############
# Read in graph and convert to dataframe
print('Reading Graph')
graph = imread(INPUT_GRAPH_FILENAME, as_gray=True)
df = pd.DataFrame(graph)

width = df.shape[1]
height = df.shape[0]

# There may be some smudges in columns that are otherwise white (e.g., col 5)
# These will be removed with thresholding (make sure a value is below a
# certain amount to consider it a line). While were at it, set all white
# columns to NaN since there is no actual graph there
print('\nFinding columns without a line')
mins = df.min(axis=0)
no_line = [mins[x] > WHITE_THRESHOLD for x in range(width)]
df[df.columns[no_line]] = np.full_like(df[df.columns[no_line]], np.nan)



# Find where the minimums occur for each column
# This method returns only the location of the first occurence of the miniumum.
# However, looking by hand at several it appears that the minimum unique
# up to maybe five pixels, although often it's completely unique. That
# translates to rough 0.5Hz error in the y-axis, which is probably small
# compared with other impacts. Therefore more complicated methods are not
# necessary.
print('Finding Line')
min_locs = df.idxmin(axis=0)

# Convert to meaningful values
# min_locs is structure as follows:
# indices -> x-axis values
# values -> y-axis values (reverse order)
print('Normalizing')
result = pd.DataFrame(columns=['x', 'y'])
result['x'] = min_locs.index
result['y'] = height - min_locs

x_norm = width / (X_MAX - X_MIN)
y_norm = height / (Y_MAX - Y_MIN)
result['x'] = (result['x'] / x_norm) + X_MIN
result['y'] = (result['y'] / y_norm) + Y_MIN

# Print some info
print('x-resolution = %1.3f' % (1 / x_norm))
print('y-resolution = %1.3f' % (1 / y_norm))

# Smooth
# The line should be somewhat continuous, but this method treats each column
# independently. This can be cleaned with smoothing. May want to play with
# window size or window type ('win_types' parameter), but this looked pretty
# good off the bat.
result_smoothed = result.rolling(WIN_SIZE, center=True, min_periods=1).mean()

# Export
result.to_csv(OUTPUT_CSV_FILENAME, index=False)
result_smoothed.to_csv(OUTPUT_CSV_FILENAME + '_smoothed', index=False)

# Plot
fig = result.plot(x='x', y='y', legend=False, xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX))
fig.get_figure().savefig('Output_Graph.png', bbox_inches='tight')

fig = result_smoothed.plot(x='x', y='y', legend=False, xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX))
fig.get_figure().savefig('Output_Graph_Smoothed.png', bbox_inches='tight')

# Finish
print('Done')
