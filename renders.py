import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.decomposition import pca

def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

def cluster_results(reduced_data, preds, centers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds cues for cluster centers and student-selected sample data
	'''

	predictions = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('Accent')

	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):   
	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i)*1.0/len(centers)), label = 'Cluster %i'%(i), s=30);
	ax.set_xlim(-10, 25)

	# Plot centers with indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);

	# Plot transformed sample points 
	ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	           s = 150, linewidth = 4, color = 'black', marker = 'x');

	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");


def site_results(reduced_data, data, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" and cues added for student-selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = data
	except:
	    print "Dataset could not be loaded. Is the file missing?"
	    return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['site'], columns = ['site'])
	# channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('Paired')

	# Color the points based on assigned Channel
	labels = ['Acme', 'Botly', 'Pinnacle', 'Sortly', 'Tabular', 'Widgetry']
	grouped = labeled.groupby('site')
	for i, channel in grouped: 
	    if i == 'Acme': 
		j = 1
	    elif i == 'Botly':
		j = 2
	    elif i == 'Pinnacle':
		j = 3
	    elif i == 'Sortly':
		j = 4
	    elif i == 'Tabular':
		j = 5
	    else:
		j = 6
	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((j-1)*1.0/6), label = labels[j-1], s=30);
	ax.set_xlim(-10, 25)
	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'site'\nTransformed Sample Data Circled");

def channel_results(reduced_data, data, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" and cues added for student-selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = data
	except:
	    print "Dataset could not be loaded. Is the file missing?"
	    return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['new_customer'], columns = ['new_customer'])
	# channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('Paired')

	# Color the points based on assigned Channel
	labels = ['Returning', 'New', 'Neither']
	grouped = labeled.groupby('new_customer')
	for i, channel in grouped: 
	    if i == 'returning': 
		j = 1
	    elif i == 'new':
		j = 2
	    else:
		j = 3
	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((j-1)*1.0/3), label = labels[j-1], s=30);
	ax.set_xlim(-10, 25)

	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'new_customer'\nTransformed Sample Data Circled");


def platform_results(reduced_data, data, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
	Data is labeled by "Channel" and cues added for student-selected sample data
	'''

	# Check that the dataset is loadable
	try:
	    full_data = data
	except:
	    print "Dataset could not be loaded. Is the file missing?"
	    return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['platform'], columns = ['platform'])
	# channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)
	
	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('Set1')

	# Color the points based on assigned Channel
	labels = ['Android', 'BlackBerry', 'ChromeOS', 'Linux', 'MacOSX', 'Macintosh', 'Other',
                 'SymbianOS', 'Unknown', 'Windows', 'WindowsPhone', 'iOS', 'iPad', 'iPhone']
	grouped = labeled.groupby('platform')
	for i, channel in grouped: 
	    if i == 'Android': 
		j = 1
	    elif i == 'BlackBerry':
		j = 2
	    elif i == 'ChromeOS':
		j = 3
	    elif i == 'Linux':
		j = 4
	    elif i == 'MacOSX':
		j = 5
	    elif i == 'Macintosh':
		j = 6
	    elif i == 'Other':
		j = 7
	    elif i == 'SymbianOS':
		j = 8
	    elif i == 'Unknown':
		j = 9
	    elif i == 'Windows':
		j = 10
	    elif i == 'WindowsPhone':
		j = 11
	    elif i == 'iOS':
		j = 12
	    elif i == 'iPad':
		j = 13
	    else:
		j = 14
	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((j-1)*1.0/14), label = labels[j-1], s=30);
	ax.set_xlim(-10, 25)

	# Plot transformed sample points   
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("PCA-Reduced Data Labeled by 'platform'\nTransformed Sample Data Circled");