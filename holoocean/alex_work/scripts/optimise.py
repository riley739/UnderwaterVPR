from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.linear_model import LinearRegression

import scipy.interpolate as interp

# Input space
xx = np.loadtxt("xx.txt")
y = np.loadtxt("y.txt")
xx = xx
y = y

# reminder: x_sparse and y_sparse are of shape [6, 7] from numpy.meshgrid
zfun_smooth_interp2d = interp.CloughTocher2DInterpolator(xx, y)   # default kind is 'linear'
# zfun_smooth_interp2d = interp.CloughTocher2DInterpolator(xx[:,0], xx[:,1], y, kind='cubic')   # default kind is 'linear'
# reminder: x_dense and y_dense are of shape (20, 21) from numpy.meshgrid
print(xx[:,0].min(), xx[:,0].max())
print(xx[:,1].min(), xx[:,1].max())
x1 = np.linspace(xx[:,0].min(), xx[:,0].max(), 100)
x2 = np.linspace(xx[:,1].min(), xx[:,1].max(), 50)
x1x2 = np.array(list(product(x1, x2)))
print(x1x2.shape)
dim1, dim2 = np.meshgrid(x1, x2)
print(dim1.shape, dim2.shape, x1[:4], x2[:4])
xvec = dim1[0,:] # 1d array of unique x values, 20 elements
yvec = dim2[:,0] # 1d array of unique y values, 21 elements
z_dense_smooth_interp2d = zfun_smooth_interp2d(x1x2[:,0], x1x2[:,1])   # output is (20, 21)-shaped array
# z_dense_smooth_interp2d = zfun_smooth_interp2d(dim1, dim2)   # output is (20, 21)-shaped array
h = plt.contourf(xvec, yvec, z_dense_smooth_interp2d.reshape((100, 50)).T)
# h = plt.contourf(xvec, yvec, z_dense_smooth_interp2d)
plt.axis('scaled')
plt.colorbar()
plt.show()

### Linear regression

# reg = LinearRegression().fit(xx, y)
# print(reg.score(xx, y))

# x1 = np.linspace(xx[:,0].min(), xx[:,0].max())
# x2 = np.linspace(xx[:,1].min(), xx[:,1].max())
# dim1, dim2 = np.meshgrid(x1, x2)
# mesh_df = np.array([xx.mean(axis=0) for i in range(dim1.size)])
# mesh_df[:,0] = dim1.ravel()
# mesh_df[:,1] = dim2.ravel()

# Z = reg.predict(mesh_df).reshape(dim1.shape)


# plt.contourf(dim1, dim2, Z)

# # zs = reg.predict(np.array([x1v.ravel(), x2v.ravel()]).T)
# # h = plt.contourf(x1v, x2v, zs)
# plt.axis('scaled')
# plt.colorbar()
# plt.show()

### Gaussian processes

# x1 = np.linspace(xx[:,0].min(), xx[:,0].max()) #p
# x2 = np.linspace(xx[:,1].min(), xx[:,1].max()) #q
# x = (np.array([x1, x2])).T

# # kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
# # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
# # kernel = RBF() # RBF kernel applies the continuity assumption
# kernel = RBF(length_scale_bounds=(1e-3, 3e1)) # RBF kernel applies the continuity assumption
# gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

# gp.fit(xx, y)

# x1x2 = np.array(list(product(x1, x2)))
# y_pred, MSE = gp.predict(x1x2, return_std=True)

# X0p, X1p = x1x2[:,0].reshape(50,50), x1x2[:,1].reshape(50,50)
# Zp = np.reshape(y_pred,(50,50))

# # alternative way to generate equivalent X0p, X1p, Zp
# # X0p, X1p = np.meshgrid(x1, x2)
# # Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]
# # Zp = np.array(Zp).T

# # fig = plt.figure(figsize=(10,8))
# # ax = fig.add_subplot(111)
# plt.contourf(X0p, X1p, Zp)
# # ax.pcolormesh(X0p, X1p, Zp)
# plt.colorbar()

# plt.show()

### Gaussian processes

# from geokrige.methods import OrdinaryKriging

# data = '''
# 1   -25.53  -48.51  4.50
# 2   -25.30  -48.49  59.00
# 3   -25.16  -48.32  40.00
# 4   -25.43  -49.26  923.50
# 5   -24.49  -49.15  360.00
# 6   -25.47  -49.46  910.00
# 7   -23.30  -49.57  512.00
# 8   -25.13  -50.01  880.00
# 9   -23.00  -50.02  450.00
# 10  -23.06  -50.21  440.00
# 11  -24.78  -50.00  1008.80
# 12  -25.27  -50.35  893.00
# 13  -24.20  -50.37  768.00
# 14  -23.16  -51.01  484.00
# 15  -22.57  -51.12  600.00
# 16  -23.54  -51.13  1020.00
# 17  -25.21  -51.30  1058.00
# 18  -23.30  -51.32  746.00
# 19  -26.29  -51.59  1100.00
# 20  -26.25  -52.21  930.00
# 21  -25.25  -52.25  880.00
# 22  -23.31  -51.13  566.00
# 23  -23.40  -51.91  542.00
# 24  -23.05  -52.26  480.00
# 25  -24.40  -52.34  540.00
# 26  -24.05  -52.36  616.40
# 27  -23.40  -52.35  530.00
# 28  -26.07  -52.41  700.00
# 29  -25.31  -53.01  513.00
# 30  -26.05  -53.04  650.00
# 31  -23.44  -53.17  480.00
# 32  -24.53  -53.33  660.00
# 33  -25.42  -53.47  400.00
# 34  -24.18  -53.55  310.00
# '''

# lines = data.strip().splitlines()
# values = [[float(x) for x in line.split()[1:]] for line in lines]
# data = np.array(values)

# # X = np.column_stack([data[:, 1], data[:, 0]])
# # yy = data[:, 2]
# # print(X.shape, yy.shape)
# # print(xx.shape, y.shape)
# # print(np.argwhere(np.isnan(xx)), np.argwhere(np.isnan(y)))
# kgn = OrdinaryKriging()
# # kgn.load(X, y)
# xx = xx[:34]
# y = y[:34]
# kgn.load(xx, y)

# kgn.variogram(plot=False)
# kgn.fit(model='exp', plot=False)

# # import geopandas as gpd
# # from geokrige.tools import TransformerGDF

# # shp_file = 'shapefile/qj614bt0216.shp'
# # prediction_gdf = gpd.read_file(shp_file).to_crs(crs='EPSG:4326')

# # transformer = TransformerGDF()
# # transformer.load(prediction_gdf)

# # meshgrid = transformer.meshgrid(density=2)
# # mask = transformer.mask()

# x1 = np.linspace(xx[:,0].min(), xx[:,0].max())
# x2 = np.linspace(xx[:,1].min(), xx[:,1].max())
# meshgrid = np.meshgrid(x1, x2)
# X, Y = meshgrid
# Z = kgn.predict(meshgrid)

# fig, ax = plt.subplots()

# cbar = ax.contourf(X, Y, Z, cmap='terrain', levels=np.arange(0, 1300, 50), extend='min')

# # Cbar aligned with the plot on the right side
# cax = fig.add_axes([0.93, 0.134, 0.02, 0.72])
# colorbar = plt.colorbar(cbar, cax=cax, orientation='vertical')

# clabels = ax.contour(X, Y, Z, levels=np.arange(250, 1501, 200), colors='k', linewidths=0.5)
# ax.clabel(clabels, fontsize=8)

# ax.grid(lw=0.2)
# ax.set_title('Elevation of Parana State (Brazil)', fontweight='bold', pad=15)
# ax.set_xlim(-55, -47.7)
# ax.set_ylim(-27, -22.2)

# plt.show()
