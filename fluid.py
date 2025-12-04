# fluid.py
import taichi as ti
import config as C
import state as S


class CubeVolume:
    # Simple axis-aligned cube volume used to spawn particles.
    def __init__(self, minimum: ti.Vector, size: ti.Vector, material_id: int):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material_id = material_id


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
        if material_id == 0:  # WATER
            S.color[i] = ti.Vector([0.2, 0.6, 1.0])
        elif material_id == 1:  # JELLY (soft or hard)
            S.color[i] = ti.Vector([1.0, 0.7, 0.75])
        elif material_id == 2:  # SNOW
            S.color[i] = ti.Vector([0.9, 0.9, 0.9])
        else:
            S.color[i] = ti.Vector([0.6, 0.6, 0.6])


def init_vols(vols):
    # Distribute all particles to a list of CubeVolume objects proportional to their physical volume.
    set_all_unused()

    total_vol = 0.0
    for vol in vols:
        total_vol += vol.volume

    next_p = 0
    for idx, vol in enumerate(vols):
        if isinstance(vol, CubeVolume):
            # number of particles in this block, proportional to volume
            par_count = int(vol.volume / total_vol * C.n_particles)
            if idx == len(vols) - 1:
                # last volume uses remaining particles
                par_count = C.n_particles - next_p
            init_cube_vol(
                next_p,
                next_p + par_count,
                vol.minimum.x,
                vol.minimum.y,
                vol.minimum.z,
                vol.size.x,
                vol.size.y,
                vol.size.z,
                vol.material_id,
            )
            next_p += par_count
        else:
            raise RuntimeError("Unknown volume type in init_vols()")
