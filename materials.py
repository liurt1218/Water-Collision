# materials.py
class MaterialConfig:
    # User-facing material configuration.
    def __init__(self, rho0: float, E: float, nu: float, kind: int, name: str = ""):
        self.rho0 = rho0
        self.E = E
        self.nu = nu
        self.kind = kind
        self.name = name or f"material_{kind}"


class MaterialRegistry:
    # Python-side registry for all materials used in the scene.
    def __init__(self):
        self.configs = []

    def register(self, cfg: MaterialConfig) -> int:
        # Register a material and return its integer id.
        mat_id = len(self.configs)
        self.configs.append(cfg)
        return mat_id


# Global registry used by the whole project
global_registry = MaterialRegistry()

# Tables used inside Taichi kernels (Python lists, read via ti.static)
rho0_table = []
E_table = []
nu_table = []
kind_table = []


def build_kernel_tables():
    # Build Python lists for kernel use, based on the global_registry.
    global rho0_table, E_table, nu_table, kind_table

    if not global_registry.configs:
        raise RuntimeError(
            "No materials registered! Register at least one MaterialConfig "
            "before calling build_kernel_tables()."
        )

    rho0_table = [cfg.rho0 for cfg in global_registry.configs]
    E_table = [cfg.E for cfg in global_registry.configs]
    nu_table = [cfg.nu for cfg in global_registry.configs]
    kind_table = [cfg.kind for cfg in global_registry.configs]

    print("[materials] Built kernel tables:")
    for idx, cfg in enumerate(global_registry.configs):
        print(
            f"  id={idx}, name={cfg.name}, kind={cfg.kind}, "
            f"rho0={cfg.rho0}, E={cfg.E}, nu={cfg.nu}"
        )
