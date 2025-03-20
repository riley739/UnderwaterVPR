import os

from PIL import Image
import plotly.graph_objects as go
import numpy as np


def calc_cam_cone_pts_3d(c2w, fov_deg, zoom = 1.0):

    fov_rad = np.deg2rad(fov_deg)

    cam_x = c2w[0, -1]
    cam_y = c2w[1, -1]
    cam_z = c2w[2, -1]

    corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn5 = [0, np.tan(fov_rad / 2.0), -1.0]

    corn1 = np.dot(c2w[:3, :3], corn1)
    corn2 = np.dot(c2w[:3, :3], corn2)
    corn3 = np.dot(c2w[:3, :3], corn3)
    corn4 = np.dot(c2w[:3, :3], corn4)
    corn5 = np.dot(c2w[:3, :3], corn5)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2) * zoom
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2) * zoom
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2) * zoom
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1] 
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2) * zoom
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]
    corn5 = np.array(corn5) / np.linalg.norm(corn5, ord=2) * zoom
    corn_x5 = cam_x + corn5[0]
    corn_y5 = cam_y + corn5[1]
    corn_z5 = cam_z + corn5[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, corn_x5]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, corn_y5]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, corn_z5]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:

    def __init__(self, poses, image_paths, colors, images=None, mesh_path=None, camera_x=1.0):
        self._fig = None

        self._camera_x = camera_x
        
        self._poses = poses
        self._image_names = image_paths
        self._colors = colors


        self.distinct_colors = [
            'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)', 'rgb(255, 255, 0)', 
            'rgb(0, 255, 255)', 'rgb(255, 0, 255)', 'rgb(128, 0, 0)', 'rgb(0, 128, 0)', 
            'rgb(0, 0, 128)', 'rgb(128, 128, 0)', 'rgb(0, 128, 128)', 'rgb(128, 0, 128)',
            'rgb(255, 165, 0)', 'rgb(0, 255, 127)', 'rgb(255, 20, 147)', 'rgb(255, 105, 180)', 
            'rgb(255, 69, 0)', 'rgb(34, 139, 34)', 'rgb(255, 0, 255)', 'rgb(135, 206, 235)',
        ]

        #TODO: Update this to = num_places
        # Repeat the color list to make 100 colors
        self.color_scale = self.distinct_colors * (1000 // len(self.distinct_colors)) 

        self._mesh = None
        if mesh_path is not None and os.path.exists(mesh_path):
            import trimesh
            self._mesh = trimesh.load(mesh_path, force='mesh')


    def update_figure(
            self, scene_bounds, 
            base_radius=0.0, zoom_scale=1.0, fov_deg=50., 
            mesh_z_shift=0.0, mesh_scale=1.0, 
            show_background=False, show_grid=False, show_ticklabels=False, y_up=False   
        ):

        fig = go.Figure()

        if self._mesh is not None:
            fig.add_trace(
                go.Mesh3d(
                    x=self._mesh.vertices[:, 0] * mesh_scale,  
                    y=self._mesh.vertices[:, 2] * -mesh_scale,  
                    z=(self._mesh.vertices[:, 1] + mesh_z_shift) * mesh_scale,  
                    i=self._mesh.faces[:, 0],
                    j=self._mesh.faces[:, 1],
                    k=self._mesh.faces[:, 2],
                    color=None,
                    facecolor=None,
                    opacity=0.8,
                    lighting={'ambient': 1},
                )
            )

        for i in range(len(self._poses)):
            
            pose = self._poses[i]
            clr = self.color_scale[self._colors[i]]
            image_name = os.path.basename(self._image_names[i])

            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5)]

            cone = calc_cam_cone_pts_3d(pose, fov_deg)
            radius = np.linalg.norm(pose[:3, -1])

            for (j, edge) in enumerate(edges):
                (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                    hovertemplate= "Place: " + str(self._colors[i]),
                    line=dict(color=clr, width=3),
                    name=image_name, ))

        # look at the center of scene
    
        #TODO Update to hover tigns
        fig.update_layout(
            height=720,
            autosize=True,
            hovermode = "closest",
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
            ),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    up=dict(x=0.0, y=0.0, z=1.0)),
                xaxis_title='X',
                yaxis_title='Z' if y_up else 'Y',
                zaxis_title='Y' if y_up else 'Z',
                xaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

        self._fig = fig
        return fig
