import matplotlib.colors as mcolors
import numpy as np

hex_colors = ["#AEC684", "#4EACB6", "#C0A3BA", "#7D82AB", "#865B96"]

rgb_colors = [mcolors.to_rgb(color) for color in hex_colors]

monet_cmap = mcolors.LinearSegmentedColormap.from_list("monet_cmap", rgb_colors, N=256)

n = 10
sample_points = np.linspace(0, 1, n)  

sampled_colors = [monet_cmap(point) for point in sample_points]
sampled_rgb = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in sampled_colors]

monet_colors = sampled_rgb

monet_colors = [(a/255, b/255, c/255) for (a,b,c) in monet_colors]

## Alternative:

hex_colors_alt = ["#1792A9", "#6FB89A", "#E7C7D4", "#BA8AB4","#854576"]

rgb_colors_alt = [mcolors.to_rgb(color) for color in hex_colors]

monet_cmap_alt = mcolors.LinearSegmentedColormap.from_list("monet_cmap_alt", rgb_colors_alt, N=256)

sample_points_alt = np.linspace(0, 1, n)  

sampled_colors_alt = [monet_cmap_alt(point) for point in sample_points_alt]
sampled_rgb_alt = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in sampled_colors_alt]

monet_colors_alt = sampled_rgb_alt

monet_colors_alt = [(a/255, b/255, c/255) for (a,b,c) in monet_colors_alt]