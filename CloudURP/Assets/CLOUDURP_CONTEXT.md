# CloudURP Project Context

This document captures the essential technical context of the volumetric cloud system so future changes can be made quickly and safely.

## 1. High-Level Architecture

Component layers:
1. FluidController (C# MonoBehaviour)
   - Orchestrates simulation + rendering each frame.
   - Owns and dispatches compute shader kernels in `FludSim.compute`.
   - Maintains RenderTexture3D resources (density, velocity, pressure, divergence).
   - Updates ray-march material (`ParticleRender.shader`) with bounds and simulation parameters.
2. Compute Shader `FludSim.compute`
   - Performs procedural density injection, advection, diffusion, velocity turbulence + wind, optional pressure projection (incompressible flow), lifecycle density decay.
3. Ray-March Shader `ParticleRender.shader`
   - Intersects view ray with volume AABB and samples `_DensityTex` to accumulate color + alpha.
4. Prebaked Source Volume
   - Created once (or rebuilt if parameters change) by injecting many spheres with noise to seed initial cloud distribution.

Frame pipeline order (current):
1. (Optional) Velocity turbulence & wind (AddVelocitySourceKernel)
2. (Optional) Velocity projection (divergence -> pressure -> subtract gradient)
3. Source density add (AddSourceKernel) [continuous seeding]
4. Diffusion iterations (DiffuseKernel) – smoothing + blending with original field
5. Advection (AdvectKernel) – transports density along velocity
6. Lifecycle decay (LifecycleKernel) – exponential fade + threshold clamp
7. Material update (sets `_DensityTex` and params)
8. GPU-driven ray marching for rendering inside `ParticleRender.shader`

## 2. Core Data Structures

RenderTexture3D allocations (size = `gridSize.x * gridSize.y * gridSize.z`):
- densityA / densityB: RFloat ping-pong for density operations.
- velocity / velocityB: ARGBFloat (xyz = velocity, w unused) + ping-pong for projection & turbulence.
- pressureA / pressureB: RFloat ping-pong for Poisson solve.
- divergence: RFloat divergence field.
- densitySource: Prebaked seed (RFloat) copied initially into densityA.

Memory note (approx): Each RFloat voxel ~4 bytes; ARGBFloat ~16 bytes.
For grid 128x64x128:
- Density pair: 128*64*128*4*2 ≈ 8 MB
- Velocity pair: 128*64*128*16*2 ≈ 32 MB
- Pressure pair: ≈ 8 MB
- Divergence: ≈ 4 MB
- Source: ≈ 4 MB
Total rough GPU usage ≈ 56 MB (excluding mip overhead / alignment).

## 3. Compute Kernels Summary

| Kernel | Purpose | Key Inputs | Key Outputs | Notes |
|--------|---------|-----------|-------------|-------|
| InitVelocityKernel | Initialize uniform velocity field | initialVelocity | velocityWrite | Called once at Awake (not every frame). |
| InjectKernel | Procedural spherical density insertion with Worley + FBM | injectPos, injectRadius, injectValue | densityWrite | Used for prebaked source & real-time injection. |
| AddSourceKernel | Adds prebaked source density gradually | sourceDensity, sourceScale | densityWrite | Caps density (maxDensity) and scales by empty space. |
| AddVelocitySourceKernel | Adds turbulent velocity + wind | velocityRead, velocitySourceScale, windDirection, windStrength | velocityWrite | Applies damping. |
| AdvectKernel | Semi-Lagrangian density transport | velocityRead, densityRead | densityWrite | Uses periodic wrap XZ, samples density with repeat sampler. |
| DiffuseKernel | Jacobi iteration blending density | bufferRead, initialBuffer, alpha, rBeta, diffusionBlend | bufferWrite | ping-pong for smoothing while preserving detail. |
| SetBoundaryKernel | Zero density edges | densityWrite | densityWrite | Not currently integrated every frame (can add). |
| LifecycleKernel | Exponential decay + threshold kill | decayRate | densityWrite | Prevents stale haze accumulation. |
| DivergenceKernel | Computes ∇·u and resets pressure | velocityRead, cellSize | divergenceWrite, pressureWrite | First pressure projection step. |
| PressureJacobiKernel | Iterative Poisson solve | pressureRead, divergenceRead, cellSize | pressureWrite | RUN multiple iterations. |
| SubtractGradientKernel | Enforces incompressibility | pressureRead, velocityRead, cellSize | velocityWrite | Produces projected velocity. |

## 4. Ray March Shader Highlights

Uniforms (material-side):
- `_DensityTex` (Texture3D) – density field.
- `_BoundsMin`, `_BoundsSize` – world-space AABB of volume mesh.
- `_GridSize` – simulation voxel dimensions (for debug if needed).
- `_CloudColor`, `_DarkColor` – base / dark shading colors.
- `_Absorption` – per-step extinction scalar.
- `_DensitySharpness` – exponent for remapping density -> crispness.
- `_Steps` – ray marching step count (quality vs performance).
- `_DebugBounds` – optional bounding box highlight.

Shading pipeline per ray sample:
1. Sample raw density.
2. Apply threshold & sharpness remap: `pow( saturate((raw - thr)/(1-thr)), _DensitySharpness )`.
3. Accumulate extinction & color (Beer-Lambert style simplification).
4. Light factor uses smoothed inverse of cumulative density for softened self-shadow look: `lightFactor = pow(1 - saturate(densityAccum * 0.15), 2)` then clamped between ~0.9 and 1.0.
5. Output alpha = `1 - transmittance` (discard if very low).

Potential improvements (future):
- Multi-scattering approximation (reduce flat lighting).
- Height-based light color tinting (sunset / ambient gradient).
- Temporal reprojection to reduce noise for high step counts.
- Shadow map or directional light integration (single-scatter). 

## 5. Parameter Behavior & Tuning Guide

Simulation:
- `gridSize`: Larger = more detail, exponential memory/time cost.
- `diffusionRate`: Higher => smoother, but erodes structure (needs balance with diffusionBlend).
- `diffusionBlend`: Interpolates original vs fully diffused result (keep ~0.4–0.6 for soft but structured clouds).
- `decayRate`: Controls lifetime; 0.3–0.7 typical for gradual fade.
- `pressureIterations`: 0 disables incompressibility; 20–40 gives stable swirling without volume collapse/expansion.
- `windStrength` + `windDirection`: Global drift; combine with turbulent velocity for natural motion.
- `velocityDamping`: Resist runaway acceleration; 0.97–0.995 for slower motion, current 0.99 baseline.

Rendering:
- `rayMarchSteps`: 48–96 balanced; >128 for cinematic stills.
- `_DensitySharpness`: Raise to get crisper edges; too high (>6) introduces popping.
- `_Absorption`: Higher darkens interior (0.4–0.8 typical). Combine with color contrast carefully.
- `_DarkColor`: Should stay relatively close to `_CloudColor` to avoid dark halos.

Source Generation:
- `cloudCount` + `cloudRadius`: Distribution and size variety; very high counts can saturate volume causing flat slabs.
- Prebaked rebuild triggers if count/radius change; uses fixed seed (deterministic) for reproducibility.

Lifecycle Interaction:
- Continuous source + decay must balance: if source adds faster than decay removes, background fog accumulates.
- Threshold in lifecycle prevents low-density persistence (value <0.001 => zero).

## 6. Extension Points

Add new scalar fields:
- Allocate another RFloat 3D RT similar to pressure/divergence.
- Pass through kernels as needed (e.g., humidity, temperature).

Add lighting improvements:
- Introduce secondary march toward light direction for self-shadowing factor per sample.
- Cache light attenuation in a low-res 3D texture for reuse.

Performance optimization ideas:
- Early ray termination already present; could also add empty-space skipping via macro AABB subdivision or a brick occupancy map.
- Reduce pressure iterations dynamically based on average divergence magnitude.
- Use half precision (RHalf / RGBAHalf) for density/velocity if banding acceptable.
- Implement temporal smoothing (reproject previous frame color based on velocity field).

## 7. Known Limitations / Outstanding Issues

- Lighting is uniform; no directional scattering or colored absorption.
- No rain / precipitation coupling (vertical velocity not constrained by buoyancy yet).
- Edge aliasing may appear with low step counts or high `_DensitySharpness`.
- Continuous source seeding can still lead to plateau density if decay is very low.
- Pressure solve uses simple Jacobi; could be replaced with more efficient methods (e.g., FFT or multigrid) for larger grids.

## 8. Safety / Debug Practices

Debugging density:
- Temporarily set `_DarkColor = (1,0,0)` to visualize high-density zones.
- Add a kernel to write out a slice to a 2D texture for inspection.

GPU resource validation:
- Always release old textures before reallocation (`Release()` called in `OnDestroy()`).
- When grid size changes at runtime, reallocate all dependent RTs (currently not implemented – future improvement).

## 9. Quick Reference: Property -> Internal Symbol Mapping

| Inspector Field | Internal Shader/CS Name | Notes |
|-----------------|-------------------------|-------|
| gridSize | gridSize (float4 xyz used) | Passed each frame. |
| diffusionRate | alpha/rBeta (derived) | Used before diffusion iterations. |
| diffusionBlend | diffusionBlend | Blends old vs diffused density. |
| decayRate | decayRate | Lifecycle exponential fade. |
| pressureIterations | iterations loop | Jacobi count. |
| cellSize | cellSize | Divergence/gradient scaling & advection normalization. |
| windDirection / windStrength | windDirection, windStrength | Turbulent velocity addition. |
| velocityDamping | velocityDamping | Applied after turbulence + wind. |
| absorption | _Absorption | In shader per-sample extinction. |
| densitySharpness | _DensitySharpness | Edge crispness remap exponent. |
| rayMarchSteps | _Steps | Defines sampling resolution. |

## 10. Adding a New Effect (Example Workflow)

Goal: Add per-voxel temperature influencing buoyancy.
1. Allocate `temperatureA`, `temperatureB` (RFloat RTs).
2. Add injection (e.g., warmer at lower altitudes) in a new `TemperatureInjectKernel`.
3. Modify `AdvectKernel` to also advect temperature (additional read/write pair).
4. Add `BuoyancyKernel` adjusting velocityWrite.y += k * (temperature - ambient).
5. Update `FluidController` to dispatch new kernels in sequence (inject -> buoyancy -> projection).
6. Optionally feed temperature into shader for color tint (pass min/max as material params).

## 11. Suggested Future Improvements (Prioritized)

1. Light/Shadow integration (single-scattering with directional light).
2. Adaptive step size (reduce samples in empty regions).
3. Runtime grid resize support & resource reallocation.
4. Temporal reprojection for smoother visuals at lower step counts.
5. Separate low-res velocity grid to cut bandwidth, keep high-res density.
6. Mipmapped density volume – sample coarser levels for far rays.

## 12. Minimal Troubleshooting Checklist

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Clouds not visible | DensityA empty / decay too high | Reduce decayRate (<0.7), verify AddSourceKernel runs. |
| Flickering | DensitySource over-injection or sharpness too high | Lower sourceScale, reduce densitySharpness. |
| Flat/blobby look | diffusionRate too high | Lower diffusionRate (<0.06), adjust diffusionBlend. |
| Stretching artifacts | Missing pressure projection | Enable `projectVelocity` and raise iterations. |
| Dark halos | `_DarkColor` too dark / absorption high | Lighten `_DarkColor`, reduce absorption. |
| Performance drop | rayMarchSteps large, grid large | Lower steps, reduce grid Y/Z or adopt half precision. |

---
Last updated: 2025-11-09.
Feel free to extend but keep sections concise and practical.
