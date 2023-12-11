import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def environment_check():
    # Initialize PyTorch3D and check system compatibility
    # 1. Verify CUDA availability for GPU acceleration
    if torch.cuda.is_available():
        print("CUDA is available")
        # Set the device to GPU
        device = torch.device("cuda")
        # Example: Create a tensor and move it to the GPU
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print("Tensor on GPU:", x)
        print()
        # 2. Confirm if PyTorch3D can utilize CUDA
        try:
            from pytorch3d import _C
            if torch.cuda.is_available():
                print("PyTorch3D is using CUDA")
            else:
                print("PyTorch3D is not using CUDA")
        except ImportError:
            print("PyTorch3D is not properly installed")
    else:
        print("CUDA is not available")
        
# Define the function for create_camera_frustum
def create_camera_frustum(camera_position, look_at=np.array([0, 0, 0], dtype=np.float64), up=np.array([0, 0, 1], dtype=np.float64), fov=60, aspect_ratio=1, near=0.2):
    # Ensure all input vectors are PyTorch tensors
    camera_position = torch.tensor(camera_position, dtype=torch.float32)
    look_at = torch.tensor(look_at, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    
    cam_dir = (look_at - camera_position).float()
    cam_dir = cam_dir / torch.norm(cam_dir)
    
    right = torch.cross(cam_dir, up).float()
    right = right / torch.norm(right)
    
    up = torch.cross(right, cam_dir)
    up = up / torch.norm(up)
    
    # Calculate half the height and width of the near plane
    near_height = 2 * torch.tan(torch.deg2rad(torch.tensor(fov)) / 2) * near
    near_width = near_height * aspect_ratio
    
    # Compute the near plane center point
    near_center = camera_position + cam_dir * near
    
    # Compute the corners of the near plane
    near_top_left = near_center + up * (near_height / 2) - right * (near_width / 2)
    near_top_right = near_center + up * (near_height / 2) + right * (near_width / 2)
    near_bottom_left = near_center - up * (near_height / 2) - right * (near_width / 2)
    near_bottom_right = near_center - up * (near_height / 2) + right * (near_width / 2)
    
    frustum_vertices = [
        camera_position.numpy(),
        near_top_left.numpy(), near_top_right.numpy(), near_bottom_left.numpy(), near_bottom_right.numpy()
    ]
    
    frustum_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 4), (4, 3), (3, 1)
    ]

    x, y, z = zip(*frustum_vertices)
    x_lines, y_lines, z_lines = [], [], []
    for p1, p2 in frustum_edges:
        x_lines.extend([x[p1], x[p2], None])
        y_lines.extend([y[p1], y[p2], None])
        z_lines.extend([z[p1], z[p2], None])

    frustum_scatter = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(
            width=2
        )
    )
    return frustum_scatter

def create_mesh_trace(verts, faces, name):
    
    # Update the vertices and faces indices
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    # Create the Mesh3d trace
    mesh_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        colorscale='Viridis',
        opacity=0.50,
        name=name
    )

    return mesh_trace

# # Test Code for create_camera_frustum
# # Define the camera parameters
# camera_position = np.array([4, -4, 4], dtype=np.float64)

# # Create the camera frustum without the far plane
# camera_frustum = create_camera_frustum(camera_position)

# # Create the plot layout
# layout = go.Layout(
#     scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='Z',
#         aspectmode='cube',
#     ),
#     margin=dict(r=0, l=0, b=0, t=0)
# )

# # Create the plot
# fig = go.Figure(data=[camera_frustum], layout=layout)

# # Show the plot
# fig.show()

def calculate_camera_position(RT):
    """
    Calculate the camera position from a rotation-translation matrix.

    Parameters:
    RT (torch.Tensor): A 4x4 rotation-translation matrix.

    Returns:
    numpy.ndarray: The calculated camera position.
    """
    rotation_matrix = RT[:3, :3].T
    translation_vector = -RT[:-1, -1].unsqueeze(1)
    camera_position = torch.matmul(rotation_matrix, translation_vector).squeeze().numpy()
    return camera_position

def rotate_verts(verts, RT):
    """
    Rotate vertices using the rotation part of the RT matrix.

    Parameters:
    verts (torch.Tensor): Vertices of the mesh.
    RT (torch.Tensor): A 4x4 rotation-translation matrix.

    Returns:
    numpy.ndarray: The rotated vertices.
    """
    rotation_matrix = RT[:3, :3].T  # Get the rotation matrix from RT
    # Apply rotation to vertices. Since verts is Nx3 and rotation_matrix is 3x3,
    # we need to transpose verts to 3xN, multiply, then transpose back to Nx3.
    rotated_verts = torch.matmul(rotation_matrix, verts.T).T
    return rotated_verts


# Define a function to add a white background to an image
def add_white_background(image):
 
    # Convert the image to RGBA mode
    image_rgba = image.convert("RGBA")
    
    # Create a new white background image of the same size as the original image, in RGBA mode
    white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    
    # Blend the original RGBA image with the white background image
    blended_image = Image.alpha_composite(white_background, image_rgba)
    
    # Convert the blended image to RGB mode
    rgb_image = blended_image.convert("RGB")
    
    return rgb_image

def create_centered_mesh(input_mesh, device):
    """
    Create a centered mesh with white vertex colors.

    Args:
    - input_mesh: A pre-loaded mesh from which vertices and faces will be extracted.
    - device: The device on which to create the mesh (e.g., 'cpu' or 'cuda').

    Returns:
    - A Meshes object with the centered mesh.
    """
    # Load the vertices and faces, ignoring textures and materials.
    verts, faces = input_mesh.detach().get_mesh_verts_faces(0)

    # Compute the center of the mesh as the average of all vertex positions.
    center = torch.mean(verts, dim=0)

    # Subtract the center from all vertex positions to move the center to the origin.
    verts = verts - center

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the mesh.
    centered_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )

    return centered_mesh

def inverse_transformation(RT):
    """Calculate the inverse of a rotation-translation matrix."""
    R_inv = RT[:3, :3].T  # Inverse of rotation matrix is its transpose
    T_inv = -torch.matmul(R_inv, RT[:-1, -1].unsqueeze(1)).squeeze()
    RT_inv = torch.eye(4)
    RT_inv[:3, :3] = R_inv
    RT_inv[:-1, -1] = T_inv
    return RT_inv

def apply_transformation(verts, RT):
    """Apply a transformation to vertices or a position."""
    return torch.matmul(RT[:3, :3], verts.T).T + RT[:-1, -1]

def rotate_verts(verts, RT):
    rotation_matrix = RT[:3, :3].T
    rotated_verts = torch.matmul(rotation_matrix, verts.T).T
    return rotated_verts

def rotate_position(position, RT):
    if isinstance(RT, np.ndarray):
        RT = torch.from_numpy(RT)
    elif isinstance(RT, list):
        RT = torch.tensor(RT, dtype=torch.float32)
    elif not isinstance(RT, torch.Tensor):
        raise TypeError("RT should be a PyTorch tensor, numpy array, or list")

    if RT.shape != (4, 4):
        raise ValueError("RT should be a 4x4 matrix")

    rotation_matrix = RT[:3, :3].T
    rotated_position = torch.matmul(rotation_matrix, torch.tensor(position, dtype=torch.float32).unsqueeze(-1)).squeeze().numpy()
    return rotated_position