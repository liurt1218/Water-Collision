# Water-Collision Based on Material Point Method

## Features:

- Basic coupling between rigid & rigid and rigid & fluid.
- Video rendering pipeline (`.png` frames and `.mp4` videos currently).
- Support JSON files for scene config. Pass in via the CLI `--scene-config YOUR_SCENE.json`.
- Collision acceleration via speeding up closest-point detection via grid based search.

## How to run:

```bash
python main.py --scene-config config/dambreak.json --sim-steps 400 --substeps 200 --out-dir dambreak
```

$\quad$ The frames will be exported to `frames/<out_dir>/frames_{%4d}.png` and the rendering result to `renderings/<out_dir>.mp4`.

## TODOs:

- Surface reconstruction.
- Add deformable objects (e.g. cloth).
- Add collision logic for walls (not only bottom). (Done)
- Read configuration files. (Done)

## Examples:

### Collision (with grid-based speedup)

![collision](examples/collision.gif)

### Dambreak

![dambreak](examples/dambreak.gif)
![dambreak_2](examples/dambreak_2.gif)

### Layering

![layering](examples/layering.gif)
![layering_2](examples/layering_2.gif)
