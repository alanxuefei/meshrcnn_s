import torch
device = torch.device("cpu")
# Assuming the necessary imports are done and device is defined
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

def display_input_image(img_input):
    """
    Display the input image.

    Parameters:
    - img_input: A PyTorch tensor representing the image to be displayed.
    """
    # Convert the PyTorch tensor to a NumPy array and transpose the axes for displaying
    img_np = img_input.cpu().detach().numpy()
    plt.imshow(img_np.transpose(1, 2, 0))
    plt.title("Input Image")
    plt.axis('off')  # Hide axis for cleaner visualization
    plt.show()

def visualize_meshes(mesh_pred):
    """
    Create visualizations for the predicted mesh.

    Parameters:
    - mesh_pred: A PyTorch3D Meshes object representing the predicted mesh.
    """
    # Get vertices and faces from the predicted mesh
    verts, faces = mesh_pred.get_mesh_verts_faces(0)
    original_verts = verts.detach().numpy()
    i, j, k = faces[:, 0].detach().numpy(), faces[:, 1].detach().numpy(), faces[:, 2].detach().numpy()

    # Define the camera view
    camera_view = dict(
        eye=dict(x=0.0, y=0.0, z=2.0),
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        projection=dict(type='perspective')
    )

    # Create subplot figures for the predicted mesh
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'surface'}]],
        subplot_titles=('Predicted Mesh',)
    )

    # Add predicted mesh to the subplot
    fig.add_trace(
        go.Mesh3d(
            x=original_verts[:, 0], 
            y=original_verts[:, 1], 
            z=original_verts[:, 2], 
            i=i, 
            j=j, 
            k=k, 
            colorscale='Viridis', 
            opacity=0.50
        )
    )

    # Update layout for the subplot
    fig.update_layout(
        autosize=False,
        width=800,  # Adjusted for a single mesh visualization
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        scene_camera=camera_view
    )
    
    # Show the subplot figure
    fig.show()
    
def visualize_mesh_with_cameras(mesh_gt, camera_poses):
    """
    Visualize the ground truth mesh along with camera frustums to indicate camera positions and orientations.
    """
    camera_view = dict(eye=dict(x=0.0, y=0.0, z=2.0), center=dict(x=0, y=0, z=0),
                       up=dict(x=0, y=1, z=0), projection=dict(type='perspective'))
    scene_layout = dict(xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-3, 3]),
                        zaxis=dict(range=[-3, 3]), camera=camera_view, aspectmode='cube')

    verts_gt, faces_gt = mesh_gt.get_mesh_verts_faces(0)
    mesh_trace_gt = go.Mesh3d(x=verts_gt[:, 0], y=verts_gt[:, 1], z=verts_gt[:, 2],
                              i=faces_gt[:, 0], j=faces_gt[:, 1], k=faces_gt[:, 2],
                              color='lightblue', opacity=0.5, name='GT Mesh')

    fig = go.Figure(data=[mesh_trace_gt])

    for index, RT in enumerate(camera_poses):
        # Decompose the RT matrix to extract the position and the forward vector.
        camera_position = RT[:3, 3].numpy()
        forward_vector = RT[:3, 2].numpy()  # Forward direction (Z-axis)
        end_point = camera_position + forward_vector * 0.3  # Scale the length of the arrow

        # Add the camera position as a dot
        fig.add_trace(go.Scatter3d(
            x=[camera_position[0]],
            y=[camera_position[1]],
            z=[camera_position[2]],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name=f'Camera {index + 1}'
        ))

        # Add an arrow to represent the orientation
        fig.add_trace(go.Scatter3d(
            x=[camera_position[0], end_point[0]],
            y=[camera_position[1], end_point[1]],
            z=[camera_position[2], end_point[2]],
            mode='lines+markers',
            marker=dict(size=2, color='red'),
            line=dict(color='red', width=2),
            showlegend=False
        ))

        # Optionally, add cones to act as arrowheads
        fig.add_trace(go.Cone(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            u=[forward_vector[0]],
            v=[forward_vector[1]],
            w=[forward_vector[2]],
            sizemode='absolute',
            sizeref=0.1,
            anchor='tip',
            showscale=False,
            colorscale=[[0, 'red'], [1, 'red']],
            name=f'Orientation {index + 1}'
        ))

    fig.update_layout(title="Ground Truth Mesh with Camera Frustums", scene=scene_layout, autosize=False,
                      width=800, height=600, margin=dict(l=50, r=50, b=50, t=50))
    fig.show()
