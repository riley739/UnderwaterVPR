# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# import numpy as np

# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)

# # Create the app
# app = pg.mkQApp("3D Wireframe Plot")

# # Create a GL View widget
# view = gl.GLViewWidget()
# view.show()
# view.setWindowTitle('PyQtGraph 3D Wireframe Example')
# view.setCameraPosition(distance=50)

# # Set background color to light blue/gray
# view.setBackgroundColor('#e6eef7')

# # Create some random data points distributed in clusters
# n_points = 40
# n_clusters = 4

# # Generate cluster centers
# cluster_centers = np.random.normal(size=(n_clusters, 3)) * 10

# # Generate points around each cluster
# points = []
# for i in range(n_clusters):
#     cluster_points = np.random.normal(size=(n_points//n_clusters, 3)) * 2 + cluster_centers[i]
#     points.append(cluster_points)

# all_points = np.vstack(points)

# # Create wireframe objects
# for i, cluster in enumerate(points):
#     for pt in cluster:
#         # Create a wireframe cube at each point
#         size = np.random.uniform(0.5, 2.0)
        
#         # Create cube vertices
#         verts = np.array([
#             [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
#             [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
#         ]) * size
        
#         # Create cube faces (each face is made of two triangles)
#         faces = np.array([
#             [0, 1, 2], [1, 3, 2],  # Front face
#             [4, 5, 6], [5, 7, 6],  # Back face
#             [0, 1, 4], [1, 5, 4],  # Top face
#             [2, 3, 6], [3, 7, 6],  # Bottom face
#             [0, 2, 4], [2, 6, 4],  # Left face
#             [1, 3, 5], [3, 7, 5]   # Right face
#         ])
        
#         # Create mesh data object
#         md = gl.MeshData(vertexes=verts, faces=faces)
        
#         # Create mesh item
#         wireframe = gl.GLMeshItem(
#             meshdata=md,
#             smooth=False,
#             drawFaces=False,
#             drawEdges=True,
#             edgeColor=(1, 0, 0, 1),  # Red color
#             pos=pt
#         )
#         view.addItem(wireframe)

# # Create a straight line of points (similar to what's in the image)
# line_points = np.zeros((15, 3))
# line_points[:, 0] = np.linspace(0, 20, 15)
# line_points[:, 1] = np.linspace(0, 20, 15)
# line_points[:, 2] = np.linspace(0, 5, 15)

# for pt in line_points:
#     # Create pyramid vertices
#     size = np.random.uniform(0.5, 1.0)
#     verts = np.array([
#         [0, 0, size],           # Tip
#         [-size, -size, -size],  # Base corner 1
#         [size, -size, -size],   # Base corner 2
#         [size, size, -size],    # Base corner 3
#         [-size, size, -size]    # Base corner 4
#     ])
    
#     # Create pyramid faces
#     faces = np.array([
#         [0, 1, 2],  # Side 1
#         [0, 2, 3],  # Side 2
#         [0, 3, 4],  # Side 3
#         [0, 4, 1],  # Side 4
#         [1, 2, 3, 4]  # Base - needs to be triangulated
#     ])
    
#     # Triangulate the base (split the quad into two triangles)
#     base_triangles = np.array([[1, 2, 3], [1, 3, 4]])
    
#     # Combine all faces
#     all_faces = np.vstack([faces[:4], base_triangles])
    
#     # Create mesh data
#     md = gl.MeshData(vertexes=verts, faces=all_faces)
    
#     wireframe = gl.GLMeshItem(
#         meshdata=md,
#         smooth=False,
#         drawFaces=False,
#         drawEdges=True,
#         edgeColor=(1, 0, 0, 1),  # Red color
#         pos=pt
#     )
#     view.addItem(wireframe)

# # Add a grid
# grid = gl.GLGridItem()
# grid.setSize(x=50, y=50, z=50)
# grid.setSpacing(x=5, y=5, z=5)
# grid.translate(0, 0, -10)
# grid.setColor((0.8, 0.8, 0.8, 0.5))  # Light gray color
# view.addItem(grid)

# # Add a simple legend using 2D plotting
# legend_plot = pg.plot()
# legend_plot.setWindowTitle('Legend')
# for i in range(30):
#     legend_plot.plot([0], [0], pen='r', name=f'00{i:03d}.png')

# # This keeps the window open
# if __name__ == '__main__':
#     pg.exec()



# # def save_cameras(dataset, log_dir):
