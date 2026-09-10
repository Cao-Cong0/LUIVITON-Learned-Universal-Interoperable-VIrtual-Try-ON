import numpy as np
from pytorch3d.io import load_obj
import argparse
import os

try:
    from .filtering import FILTER_RING_SCHEDULE, ring_for_iteration
except ImportError:
    from filtering import FILTER_RING_SCHEDULE, ring_for_iteration

# -------------------- Functions --------------------




def deformation_energy_n_ring(vertices, correspondences, target_vertices, faces, ring=1):
    """
    Compute deformation energy for a set of correspondences using an n-ring neighborhood.

    Args:
        vertices: np.ndarray of shape (n, 3), 3D coordinates of vertices.
        correspondences: np.ndarray of shape (n,), correspondence indices or -1 for no correspondence.
        target_vertices: np.ndarray of shape (n, 3), target vertices for correspondence deformation.
        faces: np.ndarray of shape (m, 3), faces of the mesh as vertex indices.
        ring: int, the number of rings to include in the neighborhood.

    Returns:
        energy: np.ndarray of shape (n,), the deformation energy for each vertex.
    """
    if ring < 1:
        raise ValueError("ring must be at least 1")
    if not isinstance(vertices, np.ndarray):
        raise TypeError(f"Expected vertices to be np.ndarray, but got {type(vertices).__name__}")
    if not isinstance(correspondences, np.ndarray):
        raise TypeError(f"Expected correspondences to be np.ndarray, but got {type(correspondences).__name__}")
    if not isinstance(target_vertices, np.ndarray):
        raise TypeError(f"Expected target_vertices to be np.ndarray, but got {type(target_vertices).__name__}")
    if not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy() if hasattr(faces, "cpu") else np.array(faces)

    n = vertices.shape[0]
    energy = np.zeros(n)

    # Build adjacency list
    adjacency_list = {i: set() for i in range(n)}
    for face in faces:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            vk = face[(i + 2) % 3]
            adjacency_list[vi].update([vj, vk])
            adjacency_list[vj].update([vi, vk])
            adjacency_list[vk].update([vi, vj])

    # # Compute n-ring adjacency
    n_ring_adjacency = {i: set(adjacency_list[i]) for i in range(n)}
    for _ in range(ring - 1):  # Expand to the desired ring
        for i in range(n):
            n_ring_adjacency[i] = n_ring_adjacency[i].union(
                *(adjacency_list[neighbor] for neighbor in n_ring_adjacency[i] if neighbor in adjacency_list)
            )


    for i in range(n):
        if correspondences[i] == -1:
            continue

        v_i = vertices[i]
        v_i_target = target_vertices[correspondences[i]]

        neighbors = list(n_ring_adjacency[i])
        valid_neighbors = [neighbor for neighbor in neighbors if correspondences[neighbor] != -1]
        if len(valid_neighbors) == 0:
            continue

        p_i = vertices[valid_neighbors] - v_i
        q_i = target_vertices[correspondences[valid_neighbors]] - v_i_target

        covariance = np.asarray(np.dot(p_i.T, q_i), dtype=np.float64)
        U, _, Vt = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        rotation = U @ correction @ Vt

        deformation = q_i - np.dot(p_i, rotation)
        energy[i] = np.sum(np.linalg.norm(deformation, axis=1) ** 2)

    return energy


def iterative_filter_correspondences(vertices, faces, correspondences, target_vertices, max_iters=4, tol=0.0):
    """
    Iteratively filter incorrect correspondences using deformation energy.
    Tracks cumulative rejections for visualization.
    """
    if not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy() if hasattr(faces, "cpu") else np.array(faces)

    if max_iters < 1 or max_iters > len(FILTER_RING_SCHEDULE):
        raise ValueError("max_iters must be between 1 and 4")

    prev_valid_count = None
    cumulative_rejected_mask = np.zeros(vertices.shape[0], dtype=bool)

    for iter_num in range(max_iters):
        ring = ring_for_iteration(iter_num)
        deformation_energy_values = deformation_energy_n_ring(
            vertices, correspondences, target_vertices, faces, ring=ring
        )

        current_valid = correspondences != -1
        valid_energies = deformation_energy_values[current_valid]
        if valid_energies.size == 0:
            raise RuntimeError("All correspondences were rejected")
        mean_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)
        threshold = mean_energy + 0.5 * std_energy

        valid_mask = current_valid & (deformation_energy_values <= threshold)
        filtered_correspondences = correspondences.copy()
        filtered_correspondences[~valid_mask] = -1  # Mark invalid correspondences as -1

        # Track cumulative rejections
        cumulative_rejected_mask |= ~valid_mask

        # Count valid correspondences
        valid_count = np.sum(filtered_correspondences != -1)
        print(
            f"Iteration {iter_num + 1}: ring={ring}, {valid_count} valid correspondences, "
            f"threshold={threshold:.4f}"
        )

        # Check for convergence
        stable = prev_valid_count == valid_count
        if tol > 0 and prev_valid_count is not None:
            stable = stable or abs(valid_count - prev_valid_count) < tol * max(valid_count, 1)
        if stable:
            print("Convergence reached.")
            break

        # Update correspondences for the next iteration
        correspondences = filtered_correspondences
        prev_valid_count = valid_count

    return filtered_correspondences, deformation_energy_values, threshold, cumulative_rejected_mask



# -------------------- Main Pipeline --------------------

def main(args):
    # Construct file paths
    target_file_path = os.path.join(args.target_file_folder, "textured.obj")
    correspondence_path = os.path.join(args.correspondence_folder, f"scan_{args.target_name}.npy")
    if args.source_mesh_path is None and args.source_file_folder is None:
        raise ValueError("Set either --source_mesh_path or --source_file_folder.")
    source_file_path = args.source_mesh_path or os.path.join(args.source_file_folder, "textured.obj")

    # Load source and target meshes
    vertices_A, faces_A, _ = load_obj(target_file_path)
    vertices_B, faces_B, _ = load_obj(source_file_path)
    faces_A = faces_A.verts_idx.cpu().numpy()
    faces_B = faces_B.verts_idx.cpu().numpy()
    vertices_A = vertices_A.cpu().numpy()
    vertices_B = vertices_B.cpu().numpy()

    # Load correspondences
    correspondences = np.load(correspondence_path) 
    # inverse_correspondences = np.load(inverse_correspondence_path) # 6890
    # import pdb; pdb.set_trace()
    # Iterative filtering
    filtered_corr, deformation_values, threshold, cumulative_rejected = iterative_filter_correspondences(
        vertices_A, faces_A, correspondences, vertices_B, max_iters=args.max_iters, tol=args.tol
    )
    print(f"Final: {np.sum(filtered_corr != -1)} valid correspondences")

    # Save the rejected correspondences
    rejected_path = os.path.join(args.correspondence_folder, f"scan_{args.target_name}_rejected.npy")
    np.save(rejected_path, cumulative_rejected)
    print(f"Rejected correspondences saved to {rejected_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteratively filter incorrect correspondences using deformation energy.")

    # Default paths
    parser.add_argument("--source_file_folder", type=str, default=None, help="Folder containing source textured.obj. Ignored when --source_mesh_path is set.")
    parser.add_argument("--source_mesh_path", type=str, default=None, help="Path to the SMPL/reference mesh used by correspondence indices.")
    parser.add_argument("--target_file_folder", type=str, required=True, help="Target SyncMVD results folder containing textured.obj.")
    parser.add_argument("--target_name", type=str, required=True, help="Name identifier for the target mesh.")
    parser.add_argument("--correspondence_folder", type=str, required=True, help="Folder containing scan_<target_name>.npy and saving scan_<target_name>_rejected.npy.")
    parser.add_argument("--max_iters", type=int, default=4, help="Filtering iterations (default and maximum: 4).")
    parser.add_argument("--tol", type=float, default=0.0, help="Optional relative count tolerance; 0 requires exact stabilization.")

    args = parser.parse_args()
    main(args)
