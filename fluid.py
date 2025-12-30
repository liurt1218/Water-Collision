# fluid.py
import taichi as ti
import config as C
import state as S
import numpy as np


class CubeVolume:
    # Simple axis-aligned cube volume used to spawn particles.
    def __init__(
        self,
        minimum: ti.Vector,
        size: ti.Vector,
        material_id: int,
        kind: int,
        color: ti.Vector,
        color_valid: int,
    ):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material_id = material_id
        self.kind = kind
        self.color_r = color[0]
        self.color_g = color[1]
        self.color_b = color[2]
        self.color_valid = color_valid


class SimpleMesh:
    def __init__(self, obj_path: str):
        verts = []
        faces = []
        with open(obj_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()[:4]
                    verts.append([float(x), float(y), float(z)])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    idx = []
                    for p in parts:
                        i = p.split("/")[0]
                        idx.append(int(i) - 1)
                    if len(idx) == 3:
                        faces.append(idx)
                    else:
                        # fan triangulation
                        for k in range(1, len(idx) - 1):
                            faces.append([idx[0], idx[k], idx[k + 1]])

        self.vertices = np.asarray(verts, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)

        if self.vertices.shape[0] == 0 or self.faces.shape[0] == 0:
            raise ValueError("No vertices or faces!")

    @property
    def bounds(self):
        vmin = self.vertices.min(axis=0)
        vmax = self.vertices.max(axis=0)
        return vmin, vmax

    @property
    def volume(self) -> float:
        v = self.vertices
        f = self.faces
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        cross01 = np.cross(v0, v1)
        vol = np.einsum("ij,ij->i", cross01, v2)
        return float(np.sum(vol) / 6.0)

    def apply_transform(self, T: np.ndarray):
        v = self.vertices
        ones = np.ones((v.shape[0], 1), dtype=v.dtype)
        vh = np.concatenate([v, ones], axis=1)
        v_trans = (T @ vh.T).T[:, :3]
        self.vertices = v_trans

    def contains(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        N = pts.shape[0]

        v = self.vertices
        f = self.faces

        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0

        dir_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        h = np.cross(dir_x[None, :], e2)
        a = np.einsum("ij,ij->i", e1, h)

        valid = np.abs(a) > 1e-12
        if not np.any(valid):
            return np.zeros(N, dtype=bool)

        v0 = v0[valid]
        e1 = e1[valid]
        e2 = e2[valid]
        h = h[valid]
        a = a[valid]
        inv_a = 1.0 / a

        inside = np.zeros(N, dtype=bool)

        for i in range(N):
            p = pts[i]

            s = p[None, :] - v0

            u = np.einsum("ij,ij->i", s, h) * inv_a
            mask = (u >= 0.0) & (u <= 1.0)
            if not np.any(mask):
                inside[i] = False
                continue

            v0_m = v0[mask]
            e1_m = e1[mask]
            e2_m = e2[mask]
            inv_a_m = inv_a[mask]
            s_m = p[None, :] - v0_m

            q = np.cross(s_m, e1_m)
            v_param = np.einsum("ij,j->i", q, dir_x) * inv_a_m
            mask2 = (v_param >= 0.0) & (u[mask] + v_param <= 1.0)
            if not np.any(mask2):
                inside[i] = False
                continue

            e2_mm = e2_m[mask2]
            q_mm = q[mask2]
            inv_a_mm = inv_a_m[mask2]

            t = np.einsum("ij,ij->i", e2_mm, q_mm) * inv_a_mm
            hits = np.count_nonzero(t > 1e-12)

            inside[i] = hits % 2 == 1

        return inside


class MeshVolume:
    def __init__(
        self,
        obj_path: str,
        minimum,
        material_id: int,
        kind: int,
        color: ti.Vector,
        color_valid: int,
        scale=None,
        size=None,
    ):
        self.material_id = material_id
        self.kind = kind
        self.color_r = float(color[0])
        self.color_g = float(color[1])
        self.color_b = float(color[2])
        self.color_valid = int(color_valid)

        self.mesh = SimpleMesh(obj_path)

        # minimum is a 3-vector (ti.Vector or list/tuple)
        min_np = np.array(
            [float(minimum[0]), float(minimum[1]), float(minimum[2])], dtype=np.float64
        )

        src_min, src_max = self.mesh.bounds
        src_size = src_max - src_min
        src_size[src_size == 0.0] = 1e-6

        if scale is not None:
            scale_np = np.array(
                [float(scale[0]), float(scale[1]), float(scale[2])], dtype=np.float64
            )
        elif size is not None:
            size_np = np.array(
                [float(size[0]), float(size[1]), float(size[2])], dtype=np.float64
            )
            scale_np = size_np / src_size
        else:
            scale_np = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[0, 0] = scale_np[0]
        T[1, 1] = scale_np[1]
        T[2, 2] = scale_np[2]
        T[0, 3] = float(min_np[0] - src_min[0] * scale_np[0])
        T[1, 3] = float(min_np[1] - src_min[1] * scale_np[1])
        T[2, 3] = float(min_np[2] - src_min[2] * scale_np[2])

        self.mesh.apply_transform(T)

        self.volume = float(abs(self.mesh.volume))
        self.bbox_min, self.bbox_max = self.mesh.bounds

    def sample_points(self, n: int) -> np.ndarray:
        batch = max(n * 4, 20000)

        rand = np.random.rand(batch, 3)
        cand = rand * (self.bbox_max - self.bbox_min)[None, :] + self.bbox_min[None, :]

        inside = self.mesh.contains(cand)
        picked = cand[inside]

        if picked.shape[0] < n:
            extra_needed = n - picked.shape[0]
            extra = self.sample_points(extra_needed)
            picked = np.concatenate([picked, extra], axis=0)

        pts = picked[:n].astype(np.float32)
        return pts


@ti.kernel
def set_all_unused():
    # Must have n_particles > 0
    for p in range(C.n_particles):
        S.is_used[p] = 0
        S.x[p] = ti.Vector([0.0, 0.0, 0.0])
        S.Jp[p] = 1.0
        S.F[p] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        S.C_apic[p] = ti.Matrix.zero(float, C.dim, C.dim)
        S.v[p] = ti.Vector.zero(float, C.dim)


@ti.kernel
def init_cube_vol(
    first_par: int,
    last_par: int,
    x_begin: float,
    y_begin: float,
    z_begin: float,
    x_size: float,
    y_size: float,
    z_size: float,
    material_id: int,
    kind: int,
    color_r: float,
    color_g: float,
    color_b: float,
    color_valid: int,
):
    # Initialize particles with random positions inside a cube volume.
    for i in range(first_par, last_par):
        rx = ti.random()
        ry = ti.random()
        rz = ti.random()
        S.x[i] = ti.Vector(
            [
                x_begin + rx * x_size,
                y_begin + ry * y_size,
                z_begin + rz * z_size,
            ]
        )
        S.Jp[i] = 1.0
        S.F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        S.v[i] = ti.Vector.zero(float, C.dim)
        S.materials[i] = material_id
        S.is_used[i] = 1
        if color_valid == 1:
            S.color[i] = ti.Vector([color_r, color_g, color_b])
        elif kind == C.WATER:  # WATER
            S.color[i] = ti.Vector([0.2, 0.6, 1.0])
        elif kind == C.JELLY:  # JELLY (soft or hard)
            S.color[i] = ti.Vector([1.0, 1.0, 0.5])
        elif kind == C.SNOW:  # SNOW
            S.color[i] = ti.Vector([0.9, 0.9, 0.9])
        else:
            S.color[i] = ti.Vector([0.6, 0.6, 0.6])


@ti.kernel
def init_range_common(
    first_par: int,
    last_par: int,
    material_id: int,
    kind: int,
    color_r: float,
    color_g: float,
    color_b: float,
    color_valid: int,
):
    for i in range(first_par, last_par):
        S.Jp[i] = 1.0
        S.F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        S.C_apic[i] = ti.Matrix.zero(float, C.dim, C.dim)
        S.v[i] = ti.Vector.zero(float, C.dim)
        S.materials[i] = material_id
        S.is_used[i] = 1

        if color_valid == 1:
            S.color[i] = ti.Vector([color_r, color_g, color_b])
        elif kind == C.WATER:  # WATER
            S.color[i] = ti.Vector([0.2, 0.6, 1.0])
        elif kind == C.JELLY:  # JELLY (soft or hard)
            S.color[i] = ti.Vector([1.0, 1.0, 0.5])
        elif kind == C.SNOW:  # SNOW
            S.color[i] = ti.Vector([0.9, 0.9, 0.9])
        else:
            S.color[i] = ti.Vector([0.6, 0.6, 0.6])


def init_vols(vols):
    # Distribute all particles to a list of CubeVolume objects proportional to their physical volume.
    set_all_unused()

    total_vol = 0.0
    for vol in vols:
        total_vol += float(vol.volume)

    next_p = 0
    for idx, vol in enumerate(vols):
        # number of particles in this block, proportional to volume
        par_count = int(float(vol.volume) / total_vol * C.n_particles)
        if idx == len(vols) - 1:
            # last volume uses remaining particles
            par_count = C.n_particles - next_p

        if par_count <= 0:
            continue

        first = next_p
        last = next_p + par_count

        if isinstance(vol, CubeVolume):
            init_cube_vol(
                first,
                last,
                vol.minimum.x,
                vol.minimum.y,
                vol.minimum.z,
                vol.size.x,
                vol.size.y,
                vol.size.z,
                vol.material_id,
                vol.kind,
                vol.color_r,
                vol.color_g,
                vol.color_b,
                vol.color_valid,
            )
            next_p += par_count

        elif isinstance(vol, MeshVolume):
            pts = vol.sample_points(par_count)
            assert pts.shape == (par_count, 3)

            x_np = S.x.to_numpy()
            x_np[first:last, :] = pts
            S.x.from_numpy(x_np)

            init_range_common(
                first,
                last,
                vol.material_id,
                vol.kind,
                vol.color_r,
                vol.color_g,
                vol.color_b,
                vol.color_valid,
            )

            next_p += par_count

        else:
            raise RuntimeError("Unknown volume type in init_vols()")
