#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define GRID_RES 32
#define MAX_CELL_VERTS 64

struct Ext
{
    float v;
    int vid;
};

__device__ __forceinline__ Ext min_ext(Ext a, Ext b) { return (b.v < a.v) ? b : a; }
__device__ __forceinline__ Ext max_ext(Ext a, Ext b) { return (b.v > a.v) ? b : a; }

// ---------------- float3 helpers ----------------
__device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 add3(const float3 &a, const float3 &b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 sub3(const float3 &a, const float3 &b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 mul3(float s, const float3 &a) { return f3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ float dot3(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(const float3 &a, const float3 &b)
{
    return f3(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 mat3_mul_rowmajor(const float *M9, const float3 &a)
{
    return f3(
        M9[0] * a.x + M9[1] * a.y + M9[2] * a.z,
        M9[3] * a.x + M9[4] * a.y + M9[5] * a.z,
        M9[6] * a.x + M9[7] * a.y + M9[8] * a.z);
}
__device__ __forceinline__ int clampi(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// Build grid kernel
__global__ void build_vertex_grid_kernel(
    const float *__restrict__ mesh_v,
    const int *__restrict__ owner,
    const int *__restrict__ rb_active,
    int *__restrict__ grid_count,
    int *__restrict__ grid_idx,
    const float *__restrict__ dmin,
    const float *__restrict__ dmax,
    int V)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= V)
        return;

    int b = owner[vid];
    if (rb_active[b] == 0)
        return;

    float x = mesh_v[vid * 3 + 0];
    float y = mesh_v[vid * 3 + 1];
    float z = mesh_v[vid * 3 + 2];

    float rx = (x - dmin[0]) / (dmax[0] - dmin[0]);
    float ry = (y - dmin[1]) / (dmax[1] - dmin[1]);
    float rz = (z - dmin[2]) / (dmax[2] - dmin[2]);

    int gx = min(max(int(rx * GRID_RES), 0), GRID_RES - 1);
    int gy = min(max(int(ry * GRID_RES), 0), GRID_RES - 1);
    int gz = min(max(int(rz * GRID_RES), 0), GRID_RES - 1);

    int cell = gx * GRID_RES * GRID_RES + gy * GRID_RES + gz;
    int base = b * (GRID_RES * GRID_RES * GRID_RES) + cell;

    int idx = atomicAdd(&grid_count[base], 1);
    if (idx < MAX_CELL_VERTS)
    {
        grid_idx[(base * MAX_CELL_VERTS) + idx] = vid;
    }
}

// Collison helper
__device__ __forceinline__ void pair_from_pid(int pid, int B, int &b1, int &b2)
{
    // offset(b1) = b1 * (2B - b1 - 1) / 2
    // find max b1 s.t. offset <= pid
    float fB = (float)B;
    float a = 1.0f;
    float bb = -(2.0f * fB - 1.0f);
    float c = 2.0f * (float)pid;
    // solve b1 ^ 2 + bb * b1 + c = 0, take smaller root then floor
    float disc = bb * bb - 4.0f * a * c;
    disc = disc < 0.f ? 0.f : disc;
    int x = (int)floorf((-(bb)-sqrtf(disc)) * 0.5f);

    // fix-up (avoid float off-by-one)
    x = clampi(x, 0, B - 2);
    auto offset = [&](int i) -> int
    {
        return (i * (2 * B - i - 1)) / 2;
    };
    while (x + 1 <= B - 2 && offset(x + 1) <= pid)
        x++;
    while (x > 0 && offset(x) > pid)
        x--;

    int off = offset(x);
    int j = pid - off; // j in [0..B-x-2]
    b1 = x;
    b2 = x + 1 + j;
}

// Collision kernel
__global__ void rigid_rigid_collisions_kernel(
    const float *__restrict__ mesh_v,
    const float *__restrict__ rb_pos,
    const float *__restrict__ rb_lv,
    const float *__restrict__ rb_av,
    const float *__restrict__ rb_Iinv,
    const float *__restrict__ rb_inv_m,
    const float *__restrict__ rb_rest,
    const float *__restrict__ rb_mu,
    const float *__restrict__ rb_halfext,
    const int *__restrict__ rb_active,
    const int *__restrict__ off,
    const int *__restrict__ cnt,
    const int *__restrict__ grid_count,
    const int *__restrict__ grid_idx,
    const float *__restrict__ dmin,
    const float *__restrict__ dmax,
    float *__restrict__ dpos,
    float *__restrict__ dlv,
    float *__restrict__ dav,
    float thresh,
    int B)
{
    int pid = (int)blockIdx.x;
    int total_pairs = B * (B - 1) / 2;
    if (pid >= total_pairs)
        return;

    int b1, b2;
    pair_from_pid(pid, B, b1, b2);
    if (rb_active[b1] == 0 || rb_active[b2] == 0)
        return;

    // broadphase sphere
    float3 c1 = f3(rb_pos[b1 * 3 + 0], rb_pos[b1 * 3 + 1], rb_pos[b1 * 3 + 2]);
    float3 c2 = f3(rb_pos[b2 * 3 + 0], rb_pos[b2 * 3 + 1], rb_pos[b2 * 3 + 2]);
    float3 he1 = f3(rb_halfext[b1 * 3 + 0], rb_halfext[b1 * 3 + 1], rb_halfext[b1 * 3 + 2]);
    float3 he2 = f3(rb_halfext[b2 * 3 + 0], rb_halfext[b2 * 3 + 1], rb_halfext[b2 * 3 + 2]);
    float r1b = sqrtf(dot3(he1, he1));
    float r2b = sqrtf(dot3(he2, he2));
    float3 dc = sub3(c2, c1);
    float dist2 = dot3(dc, dc);
    float rad = r1b + r2b + thresh;
    if (dist2 > rad * rad)
        return;

    // domain ext
    float domx = dmax[0] - dmin[0];
    float domy = dmax[1] - dmin[1];
    float domz = dmax[2] - dmin[2];
    float inv_domx = 1.0f / (domx + 1e-12f);
    float inv_domy = 1.0f / (domy + 1e-12f);
    float inv_domz = 1.0f / (domz + 1e-12f);

    // search nearest (vertex-vertex) using grid of body b2
    float best_d2 = 1e30f;
    int best1 = -1, best2 = -1;

    int start1 = off[b1];
    int count1 = cnt[b1];

    const int CELLS = GRID_RES * GRID_RES * GRID_RES;
    const int base_b2 = b2 * CELLS;

    for (int k1 = threadIdx.x; k1 < count1; k1 += blockDim.x)
    {
        int vid1 = start1 + k1;
        float3 x1 = f3(mesh_v[vid1 * 3 + 0], mesh_v[vid1 * 3 + 1], mesh_v[vid1 * 3 + 2]);

        int gx0 = clampi((int)(((x1.x - dmin[0]) * inv_domx) * GRID_RES), 0, GRID_RES - 1);
        int gy0 = clampi((int)(((x1.y - dmin[1]) * inv_domy) * GRID_RES), 0, GRID_RES - 1);
        int gz0 = clampi((int)(((x1.z - dmin[2]) * inv_domz) * GRID_RES), 0, GRID_RES - 1);

        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    int gx = gx0 + dx, gy = gy0 + dy, gz = gz0 + dz;
                    if (gx < 0 || gy < 0 || gz < 0 || gx >= GRID_RES || gy >= GRID_RES || gz >= GRID_RES)
                        continue;
                    int cell = gx * GRID_RES * GRID_RES + gy * GRID_RES + gz;
                    int base = base_b2 + cell;

                    int cnum = grid_count[base];
                    cnum = cnum > MAX_CELL_VERTS ? MAX_CELL_VERTS : cnum;
                    for (int ci = 0; ci < cnum; ci++)
                    {
                        int vid2 = grid_idx[(base * MAX_CELL_VERTS) + ci];
                        float3 x2 = f3(mesh_v[vid2 * 3 + 0], mesh_v[vid2 * 3 + 1], mesh_v[vid2 * 3 + 2]);
                        float3 d = sub3(x2, x1);
                        float d2 = dot3(d, d);
                        if (d2 < best_d2)
                        {
                            best_d2 = d2;
                            best1 = vid1;
                            best2 = vid2;
                        }
                    }
                }
    }

    // block reduction
    extern __shared__ unsigned char smem[];
    float *sh_d2 = (float *)smem;
    int *sh_v1 = (int *)(sh_d2 + blockDim.x);
    int *sh_v2 = (int *)(sh_v1 + blockDim.x);

    int tid = threadIdx.x;
    sh_d2[tid] = best_d2;
    sh_v1[tid] = best1;
    sh_v2[tid] = best2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sh_d2[tid + s] < sh_d2[tid])
            {
                sh_d2[tid] = sh_d2[tid + s];
                sh_v1[tid] = sh_v1[tid + s];
                sh_v2[tid] = sh_v2[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid != 0)
        return;
    if (sh_v1[0] < 0)
        return;

    float min_dist = sqrtf(sh_d2[0]);
    if (!(min_dist < thresh && min_dist > 1e-6f))
        return;

    int vid1 = sh_v1[0];
    int vid2 = sh_v2[0];

    float3 p1 = f3(mesh_v[vid1 * 3 + 0], mesh_v[vid1 * 3 + 1], mesh_v[vid1 * 3 + 2]);
    float3 p2 = f3(mesh_v[vid2 * 3 + 0], mesh_v[vid2 * 3 + 1], mesh_v[vid2 * 3 + 2]);
    float3 n = mul3(1.0f / min_dist, sub3(p2, p1));

    float penetration = thresh - min_dist;

    float invm1 = rb_inv_m[b1];
    float invm2 = rb_inv_m[b2];
    float wsum = invm1 + invm2 + 1e-6f;

    float3 corr1 = mul3(penetration * (invm1 / wsum), n);
    float3 corr2 = mul3(penetration * (invm2 / wsum), n);

    // accumulate dpos
    atomicAdd(&dpos[b1 * 3 + 0], -corr1.x);
    atomicAdd(&dpos[b1 * 3 + 1], -corr1.y);
    atomicAdd(&dpos[b1 * 3 + 2], -corr1.z);
    atomicAdd(&dpos[b2 * 3 + 0], corr2.x);
    atomicAdd(&dpos[b2 * 3 + 1], corr2.y);
    atomicAdd(&dpos[b2 * 3 + 2], corr2.z);

    // use "corrected" centers for r1/r2 like Taichi (best effort in parallel)
    float3 c1c = sub3(c1, corr1);
    float3 c2c = add3(c2, corr2);

    float3 r1 = sub3(p1, c1c);
    float3 r2 = sub3(p2, c2c);

    // ===== normal impulse + friction impulse (Taichi-like) =====
    float3 v1 = f3(rb_lv[b1 * 3 + 0], rb_lv[b1 * 3 + 1], rb_lv[b1 * 3 + 2]);
    float3 v2 = f3(rb_lv[b2 * 3 + 0], rb_lv[b2 * 3 + 1], rb_lv[b2 * 3 + 2]);
    float3 w1 = f3(rb_av[b1 * 3 + 0], rb_av[b1 * 3 + 1], rb_av[b1 * 3 + 2]);
    float3 w2 = f3(rb_av[b2 * 3 + 0], rb_av[b2 * 3 + 1], rb_av[b2 * 3 + 2]);

    float3 v1c = add3(v1, cross3(w1, r1));
    float3 v2c = add3(v2, cross3(w2, r2));
    float3 vrel = sub3(v2c, v1c);
    float vrel_n = dot3(vrel, n);

    if (vrel_n < 0.0f)
    {
        const float *I1 = rb_Iinv + b1 * 9;
        const float *I2 = rb_Iinv + b2 * 9;

        float e = 0.5f * (rb_rest[b1] + rb_rest[b2]);
        float mu = 0.5f * (rb_mu[b1] + rb_mu[b2]);

        float3 rn1 = cross3(r1, n);
        float3 rn2 = cross3(r2, n);

        float ang1 = dot3(cross3(mat3_mul_rowmajor(I1, rn1), r1), n);
        float ang2 = dot3(cross3(mat3_mul_rowmajor(I2, rn2), r2), n);

        float denom_n = invm1 + invm2 + ang1 + ang2 + 1e-6f;
        float jn = -(1.0f + e) * vrel_n / denom_n;
        float3 Jn = mul3(jn, n);

        // linear delta-v
        atomicAdd(&dlv[b1 * 3 + 0], -invm1 * Jn.x);
        atomicAdd(&dlv[b1 * 3 + 1], -invm1 * Jn.y);
        atomicAdd(&dlv[b1 * 3 + 2], -invm1 * Jn.z);
        atomicAdd(&dlv[b2 * 3 + 0], invm2 * Jn.x);
        atomicAdd(&dlv[b2 * 3 + 1], invm2 * Jn.y);
        atomicAdd(&dlv[b2 * 3 + 2], invm2 * Jn.z);

        // angular delta-w
        float3 dw1 = mat3_mul_rowmajor(I1, cross3(r1, mul3(-1.0f, Jn)));
        float3 dw2 = mat3_mul_rowmajor(I2, cross3(r2, Jn));

        atomicAdd(&dav[b1 * 3 + 0], dw1.x);
        atomicAdd(&dav[b1 * 3 + 1], dw1.y);
        atomicAdd(&dav[b1 * 3 + 2], dw1.z);
        atomicAdd(&dav[b2 * 3 + 0], dw2.x);
        atomicAdd(&dav[b2 * 3 + 1], dw2.y);
        atomicAdd(&dav[b2 * 3 + 2], dw2.z);

        // friction impulse (Taichi-style using same denom form)
        // recompute tangential direction using current vrel (Jacobi approx)
        float ncomp = dot3(vrel, n);
        float3 vtan = sub3(vrel, mul3(ncomp, n));
        float vt2 = dot3(vtan, vtan);
        if (vt2 > 1e-12f)
        {
            float vt = sqrtf(vt2);
            float3 t = mul3(1.0f / vt, vtan);

            float3 rt1 = cross3(r1, t);
            float3 rt2 = cross3(r2, t);

            float ang1t = dot3(cross3(mat3_mul_rowmajor(I1, rt1), r1), t);
            float ang2t = dot3(cross3(mat3_mul_rowmajor(I2, rt2), r2), t);

            float denom_t = invm1 + invm2 + ang1t + ang2t + 1e-6f;

            float jt_raw = -vt / denom_t;
            float max_jt = mu * fabsf(jn);
            float jt = jt_raw;
            if (jt > max_jt)
                jt = max_jt;
            if (jt < -max_jt)
                jt = -max_jt;

            float3 Jt = mul3(jt, t);

            atomicAdd(&dlv[b1 * 3 + 0], -invm1 * Jt.x);
            atomicAdd(&dlv[b1 * 3 + 1], -invm1 * Jt.y);
            atomicAdd(&dlv[b1 * 3 + 2], -invm1 * Jt.z);
            atomicAdd(&dlv[b2 * 3 + 0], invm2 * Jt.x);
            atomicAdd(&dlv[b2 * 3 + 1], invm2 * Jt.y);
            atomicAdd(&dlv[b2 * 3 + 2], invm2 * Jt.z);

            float3 dwt1 = mat3_mul_rowmajor(I1, cross3(r1, mul3(-1.0f, Jt)));
            float3 dwt2 = mat3_mul_rowmajor(I2, cross3(r2, Jt));

            atomicAdd(&dav[b1 * 3 + 0], dwt1.x);
            atomicAdd(&dav[b1 * 3 + 1], dwt1.y);
            atomicAdd(&dav[b1 * 3 + 2], dwt1.z);
            atomicAdd(&dav[b2 * 3 + 0], dwt2.x);
            atomicAdd(&dav[b2 * 3 + 1], dwt2.y);
            atomicAdd(&dav[b2 * 3 + 2], dwt2.z);
        }
    }
}

// Apply deltas
__global__ void apply_rigid_deltas_kernel(
    float *rb_pos, float *rb_lv, float *rb_av,
    const float *dpos, const float *dlv, const float *dav,
    int B)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B)
        return;
    for (int i = 0; i < 3; i++)
    {
        rb_pos[b * 3 + i] += dpos[b * 3 + i];
        rb_lv[b * 3 + i] += dlv[b * 3 + i];
        rb_av[b * 3 + i] += dav[b * 3 + i];
    }
}

// Collision wrapper
void build_vertex_grid_cuda(
    torch::Tensor mesh_vertices,
    torch::Tensor vertex_owner,
    torch::Tensor rb_active,
    torch::Tensor grid_count,
    torch::Tensor grid_indices,
    torch::Tensor domain_min,
    torch::Tensor domain_max)
{
    int V = mesh_vertices.size(0);
    int threads = 256, blocks = (V + threads - 1) / threads;
    build_vertex_grid_kernel<<<blocks, threads>>>(
        mesh_vertices.data_ptr<float>(),
        vertex_owner.data_ptr<int>(),
        rb_active.data_ptr<int>(),
        grid_count.data_ptr<int>(),
        grid_indices.data_ptr<int>(),
        domain_min.data_ptr<float>(),
        domain_max.data_ptr<float>(),
        V);
}

void rigid_rigid_collisions_cuda(
    torch::Tensor mesh_vertices,
    torch::Tensor rb_pos,
    torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel,
    torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass,
    torch::Tensor rb_restitution,
    torch::Tensor rb_friction,
    torch::Tensor rb_half_extents,
    torch::Tensor rb_active,
    torch::Tensor rb_mesh_vert_offset,
    torch::Tensor rb_mesh_vert_count,
    torch::Tensor grid_count,
    torch::Tensor grid_indices,
    torch::Tensor domain_min,
    torch::Tensor domain_max,
    torch::Tensor dpos,
    torch::Tensor dlv,
    torch::Tensor dav,
    float thresh)
{
    int B = (int)rb_pos.size(0);
    int total_pairs = B * (B - 1) / 2;
    if (total_pairs <= 0)
        return;

    const int threads = 128;
    size_t shmem = threads * (sizeof(float) + 2 * sizeof(int));

    rigid_rigid_collisions_kernel<<<total_pairs, threads, shmem>>>(
        mesh_vertices.data_ptr<float>(),
        rb_pos.data_ptr<float>(),
        rb_lin_vel.data_ptr<float>(),
        rb_ang_vel.data_ptr<float>(),
        rb_inv_inertia_world.data_ptr<float>(),
        rb_inv_mass.data_ptr<float>(),
        rb_restitution.data_ptr<float>(),
        rb_friction.data_ptr<float>(),
        rb_half_extents.data_ptr<float>(),
        rb_active.data_ptr<int>(),
        rb_mesh_vert_offset.data_ptr<int>(),
        rb_mesh_vert_count.data_ptr<int>(),
        grid_count.data_ptr<int>(),
        grid_indices.data_ptr<int>(),
        domain_min.data_ptr<float>(),
        domain_max.data_ptr<float>(),
        dpos.data_ptr<float>(),
        dlv.data_ptr<float>(),
        dav.data_ptr<float>(),
        thresh,
        B);
}

void apply_rigid_deltas_cuda(
    torch::Tensor rb_pos, torch::Tensor rb_lv, torch::Tensor rb_av,
    torch::Tensor dpos, torch::Tensor dlv, torch::Tensor dav)
{
    int B = rb_pos.size(0);
    int threads = 256, blocks = (B + threads - 1) / threads;
    apply_rigid_deltas_kernel<<<blocks, threads>>>(
        rb_pos.data_ptr<float>(),
        rb_lv.data_ptr<float>(),
        rb_av.data_ptr<float>(),
        dpos.data_ptr<float>(),
        dlv.data_ptr<float>(),
        dav.data_ptr<float>(),
        B);
}

// Domain wall collision kernel
__global__ void handle_domain_walls_kernel(
    const float *__restrict__ mesh_v,   // (V,3)
    float *__restrict__ rb_pos,         // (B,3) in/out
    float *__restrict__ rb_lv,          // (B,3) in/out
    float *__restrict__ rb_av,          // (B,3) in/out
    const float *__restrict__ rb_Iinv,  // (B,9) row-major
    const float *__restrict__ rb_inv_m, // (B,)
    const float *__restrict__ rb_rest,  // (B,)
    const float *__restrict__ rb_mu,    // (B,)
    const int *__restrict__ rb_active,  // (B,)
    const int *__restrict__ off,        // (B,)
    const int *__restrict__ cnt,        // (B,)
    const float *__restrict__ dmin,     // (3,)
    const float *__restrict__ dmax,     // (3,)
    float max_pen)
{
    int b = blockIdx.x;
    if (rb_active[b] == 0)
        return;

    int start = off[b];
    int count = cnt[b];

    // thread-local extrema
    Ext minx{1e30f, -1}, maxx{-1e30f, -1};
    Ext miny{1e30f, -1}, maxy{-1e30f, -1};
    Ext minz{1e30f, -1}, maxz{-1e30f, -1};

    for (int k = threadIdx.x; k < count; k += blockDim.x)
    {
        int vid = start + k;
        float x = mesh_v[vid * 3 + 0];
        float y = mesh_v[vid * 3 + 1];
        float z = mesh_v[vid * 3 + 2];
        minx = min_ext(minx, {x, vid});
        maxx = max_ext(maxx, {x, vid});
        miny = min_ext(miny, {y, vid});
        maxy = max_ext(maxy, {y, vid});
        minz = min_ext(minz, {z, vid});
        maxz = max_ext(maxz, {z, vid});
    }

    // dynamic shared: 6 arrays of Ext, each length blockDim.x
    extern __shared__ Ext sh[];
    Ext *sh_minx = sh + 0 * blockDim.x;
    Ext *sh_maxx = sh + 1 * blockDim.x;
    Ext *sh_miny = sh + 2 * blockDim.x;
    Ext *sh_maxy = sh + 3 * blockDim.x;
    Ext *sh_minz = sh + 4 * blockDim.x;
    Ext *sh_maxz = sh + 5 * blockDim.x;

    int t = threadIdx.x;
    sh_minx[t] = minx;
    sh_maxx[t] = maxx;
    sh_miny[t] = miny;
    sh_maxy[t] = maxy;
    sh_minz[t] = minz;
    sh_maxz[t] = maxz;
    __syncthreads();

    // reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            sh_minx[t] = min_ext(sh_minx[t], sh_minx[t + stride]);
            sh_maxx[t] = max_ext(sh_maxx[t], sh_maxx[t + stride]);
            sh_miny[t] = min_ext(sh_miny[t], sh_miny[t + stride]);
            sh_maxy[t] = max_ext(sh_maxy[t], sh_maxy[t + stride]);
            sh_minz[t] = min_ext(sh_minz[t], sh_minz[t + stride]);
            sh_maxz[t] = max_ext(sh_maxz[t], sh_maxz[t + stride]);
        }
        __syncthreads();
    }

    if (t != 0)
        return;

    Ext s_minx = sh_minx[0], s_maxx = sh_maxx[0];
    Ext s_miny = sh_miny[0], s_maxy = sh_maxy[0];
    Ext s_minz = sh_minz[0], s_maxz = sh_maxz[0];

    float3 c = f3(rb_pos[b * 3 + 0], rb_pos[b * 3 + 1], rb_pos[b * 3 + 2]);
    float3 v = f3(rb_lv[b * 3 + 0], rb_lv[b * 3 + 1], rb_lv[b * 3 + 2]);
    float3 w = f3(rb_av[b * 3 + 0], rb_av[b * 3 + 1], rb_av[b * 3 + 2]);

    const float inv_m = rb_inv_m[b];
    const float rest = rb_rest[b];
    const float mu = rb_mu[b];
    const float *I = rb_Iinv + b * 9;

    auto apply_wall = [&](Ext e, const float3 &n, float penetration)
    {
        if (e.vid < 0)
            return;
        if (penetration <= 0.f)
            return;
        penetration = fminf(penetration, max_pen);

        float3 cp = f3(mesh_v[e.vid * 3 + 0], mesh_v[e.vid * 3 + 1], mesh_v[e.vid * 3 + 2]);

        // push out
        c = add3(c, mul3(penetration, n));
        cp = add3(cp, mul3(penetration, n));

        float3 r = sub3(cp, c);
        float3 v_c = add3(v, cross3(w, r));
        float vrel_n = dot3(v_c, n);

        float j_n = 0.f;
        if (vrel_n < 0.f)
        {
            float3 rn = cross3(r, n);
            float3 I_rn = mat3_mul_rowmajor(I, rn);
            float ang = dot3(cross3(I_rn, r), n);
            float denom = inv_m + ang + 1e-6f;

            j_n = -(1.f + rest) * vrel_n / denom;
            float3 Jn = mul3(j_n, n);

            v = add3(v, mul3(inv_m, Jn));
            w = add3(w, mat3_mul_rowmajor(I, cross3(r, Jn)));
        }

        // friction
        float3 v_c2 = add3(v, cross3(w, r));
        float ncomp = dot3(v_c2, n);
        float3 v_tan = sub3(v_c2, mul3(ncomp, n));
        float vt2 = dot3(v_tan, v_tan);

        if (vt2 > 1e-12f)
        {
            float vt = sqrtf(vt2);
            float3 tdir = mul3(1.f / vt, v_tan);

            float3 rt = cross3(r, tdir);
            float3 I_rt = mat3_mul_rowmajor(I, rt);
            float ang_t = dot3(cross3(I_rt, r), tdir);
            float denom_t = inv_m + ang_t + 1e-6f;

            float j_t = -vt / denom_t;
            float max_jt = mu * fabsf(j_n);
            if (j_t > max_jt)
                j_t = max_jt;
            if (j_t < -max_jt)
                j_t = -max_jt;

            float3 Jt = mul3(j_t, tdir);
            v = add3(v, mul3(inv_m, Jt));
            w = add3(w, mat3_mul_rowmajor(I, cross3(r, Jt)));
        }
    };

    apply_wall(s_miny, f3(0, 1, 0), dmin[1] - s_miny.v);
    apply_wall(s_maxy, f3(0, -1, 0), s_maxy.v - dmax[1]);
    apply_wall(s_minx, f3(1, 0, 0), dmin[0] - s_minx.v);
    apply_wall(s_maxx, f3(-1, 0, 0), s_maxx.v - dmax[0]);
    apply_wall(s_minz, f3(0, 0, 1), dmin[2] - s_minz.v);
    apply_wall(s_maxz, f3(0, 0, -1), s_maxz.v - dmax[2]);

    rb_pos[b * 3 + 0] = c.x;
    rb_pos[b * 3 + 1] = c.y;
    rb_pos[b * 3 + 2] = c.z;
    rb_lv[b * 3 + 0] = v.x;
    rb_lv[b * 3 + 1] = v.y;
    rb_lv[b * 3 + 2] = v.z;
    rb_av[b * 3 + 0] = w.x;
    rb_av[b * 3 + 1] = w.y;
    rb_av[b * 3 + 2] = w.z;
}

// Domain wall wrapper
void handle_domain_walls_cuda(
    torch::Tensor mesh_vertices,
    torch::Tensor rb_pos,
    torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel,
    torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass,
    torch::Tensor rb_restitution,
    torch::Tensor rb_friction,
    torch::Tensor rb_active,
    torch::Tensor rb_mesh_vert_offset,
    torch::Tensor rb_mesh_vert_count,
    torch::Tensor domain_min,
    torch::Tensor domain_max,
    float max_pen)
{
    int B = (int)rb_pos.size(0);
    int threads = 256;
    size_t shmem = (size_t)(6 * threads) * sizeof(Ext);

    handle_domain_walls_kernel<<<B, threads, shmem>>>(
        (float *)mesh_vertices.data_ptr<float>(),
        (float *)rb_pos.data_ptr<float>(),
        (float *)rb_lin_vel.data_ptr<float>(),
        (float *)rb_ang_vel.data_ptr<float>(),
        (float *)rb_inv_inertia_world.data_ptr<float>(),
        (float *)rb_inv_mass.data_ptr<float>(),
        (float *)rb_restitution.data_ptr<float>(),
        (float *)rb_friction.data_ptr<float>(),
        (int *)rb_active.data_ptr<int>(),
        (int *)rb_mesh_vert_offset.data_ptr<int>(),
        (int *)rb_mesh_vert_count.data_ptr<int>(),
        (float *)domain_min.data_ptr<float>(),
        (float *)domain_max.data_ptr<float>(),
        max_pen);
}

// Update mesh kernel
__global__ void update_mesh_kernel(
    float *__restrict__ mesh_vertices,
    float *__restrict__ mesh_normals,
    const float *__restrict__ mesh_local_v,
    const float *__restrict__ mesh_local_n,
    const float *__restrict__ rb_pos,
    const float *__restrict__ rb_rot,
    const int *__restrict__ owner,
    int V)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= V)
        return;

    int b = owner[vid];

    float px = rb_pos[b * 3 + 0];
    float py = rb_pos[b * 3 + 1];
    float pz = rb_pos[b * 3 + 2];

    const float *R = rb_rot + b * 9; // row-major

    float lx = mesh_local_v[vid * 3 + 0];
    float ly = mesh_local_v[vid * 3 + 1];
    float lz = mesh_local_v[vid * 3 + 2];

    float wx = px + (R[0] * lx + R[1] * ly + R[2] * lz);
    float wy = py + (R[3] * lx + R[4] * ly + R[5] * lz);
    float wz = pz + (R[6] * lx + R[7] * ly + R[8] * lz);

    mesh_vertices[vid * 3 + 0] = wx;
    mesh_vertices[vid * 3 + 1] = wy;
    mesh_vertices[vid * 3 + 2] = wz;

    float nx = mesh_local_n[vid * 3 + 0];
    float ny = mesh_local_n[vid * 3 + 1];
    float nz = mesh_local_n[vid * 3 + 2];

    float nwx = (R[0] * nx + R[1] * ny + R[2] * nz);
    float nwy = (R[3] * nx + R[4] * ny + R[5] * nz);
    float nwz = (R[6] * nx + R[7] * ny + R[8] * nz);

    mesh_normals[vid * 3 + 0] = nwx;
    mesh_normals[vid * 3 + 1] = nwy;
    mesh_normals[vid * 3 + 2] = nwz;
}

// Update wrapper
void update_all_mesh_vertices_cuda(
    torch::Tensor mesh_vertices,
    torch::Tensor mesh_normals,
    torch::Tensor mesh_local_vertices,
    torch::Tensor mesh_local_normals,
    torch::Tensor rb_pos,
    torch::Tensor rb_rot,
    torch::Tensor vertex_owner)
{
    const int V = (int)mesh_vertices.size(0);
    const int threads = 256;
    const int blocks = (V + threads - 1) / threads;

    update_mesh_kernel<<<blocks, threads>>>(
        (float *)mesh_vertices.data_ptr<float>(),
        (float *)mesh_normals.data_ptr<float>(),
        (const float *)mesh_local_vertices.data_ptr<float>(),
        (const float *)mesh_local_normals.data_ptr<float>(),
        (const float *)rb_pos.data_ptr<float>(),
        (const float *)rb_rot.data_ptr<float>(),
        (const int *)vertex_owner.data_ptr<int>(),
        V);
}
