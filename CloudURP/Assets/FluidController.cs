// url parameter omitted because this is an edited file suggestion
using TreeEditor;
using UnityEngine;
using UnityEngine.Rendering;

// FluidController: orchestrates 3D density simulation + ray-march rendering
// Attach to a GameObject (e.g. the volume cube) and assign:
// - computeShader = FludSim.compute
// - rayMarchMaterial = material using Custom/RayMarchShader
// - volumeRenderer = MeshRenderer of the cube
public class FluidController : MonoBehaviour
{
  [Header("Simulation Settings")]
  public ComputeShader computeShader;

  public Vector3Int gridSize = new Vector3Int( 128, 64, 128);

  [Tooltip("Constant diffusion rate (small value ~0.02 - 0.1).")]
  public float diffusionRate = 0.05f;
  [Tooltip("Jacobi iterations for diffusion (more = smoother but slower).")]
  public int diffusionIterations = 15;

  [Tooltip("Density decay rate over time (0 = no decay, 1 = fast decay).")]
  [Range(0f, 1f)]
  public float decayRate = 0.5f;

  [Header("Pressure Projection")]
  public bool projectVelocity = true;
  [Tooltip("Jacobi iterations for pressure solve (divergence-free enforcement).")]
  [Range(0, 100)]
  public int pressureIterations = 30;
  [Tooltip("Grid cell size used for divergence/gradient (world units).")]
  public float cellSize = 1f;

  [Header("Velocity Init / Sources")]
  public Vector3 initialVelocity = new Vector3(0, 0.25f, 0);


  [Header("Generic Source (Prebaked)")]
  public bool addConstantSource = true;
  public float sourceScale = 1.0f;

  [Tooltip("Only add source where density is low (prevents infinite accumulation)")]
  public bool smartSourceInjection = true;

  [Tooltip("Time in seconds for source injection cycle (high = slower cycle)")]
  [Range(5f, 60f)]
  public float sourceCycleTime = 20f;

  private float sourceTimer = 0f;

  [Tooltip("Number of cloud spheres to generate")]
  [Range(1, 500)]
  public int cloudCount = 99;

  [Tooltip("Radius of each cloud sphere")]
  [Range(5f, 40f)]
  public float cloudRadius = 15f;

  [Header("Rendering")]
  public Material rayMarchMaterial;
  public MeshRenderer volumeRenderer;
  [Range(0.1f, 5.0f)]
  public float densitySharpness = 1.0f;
  [Range(4, 256)]
  public int rayMarchSteps = 64;
  public Color cloudColor = new Color(1.0f, 1.0f, 0.98f);
  public Color darkCloudColor = new Color(0.75f, 0.75f, 0.75f); // Mais claro para evitar bordas pretas

  [Range(0f, 2f)]
  public float absorption = 0.5f;

  [Space]
  [Tooltip("Controls light scattering. 0 = uniform, 0.7 = strong forward scatter (silver lining)")]
  [Range(-0.9f, 0.9f)]
  public float gAnisotropy = 0.4f;

  public bool debugBounds = false;

  // Internal RenderTextures
  private RenderTexture densityA, densityB;
  private RenderTexture velocity, velocityB; // <- added velocityB
  private RenderTexture pressureA, pressureB;
  private RenderTexture divergence;

  // Optional source density texture
  private RenderTexture densitySource;

  // Kernels
  private int kInitVelocity = -1;
  private int kInject = -1;
  private int kAddSource = -1;
  private int kAddVelocitySource = -1;
  private int kAdvect = -1;
  private int kDiffuse = -1;
  private int kLifecycle = -1;
  private int kDivergence = -1;
  private int kPressureJacobi = -1;
  private int kSubtractGradient = -1;

  // Cached bounds
  private Vector3 boundsMin;
  private Vector3 boundsSize;

  // Cached source settings to rebuild prebaked volume when changed at runtime
  private int cachedCloudCount;
  private float cachedCloudRadius;

  // ------------------------------------------------------
  // Unity Lifecycle
  // ------------------------------------------------------
  void Awake()
  {
    if (!computeShader)
    {
      Debug.LogError("ComputeShader not assigned.", this);
      enabled = false;
      return;
    }

    Allocate3DTexture(ref densityA, RenderTextureFormat.RFloat);
    Allocate3DTexture(ref densityB, RenderTextureFormat.RFloat);
    Allocate3DTexture(ref velocity, RenderTextureFormat.ARGBFloat);
    Allocate3DTexture(ref velocityB, RenderTextureFormat.ARGBFloat); // <- allocate velocityB
    Allocate3DTexture(ref pressureA, RenderTextureFormat.RFloat);
    Allocate3DTexture(ref pressureB, RenderTextureFormat.RFloat);
    Allocate3DTexture(ref divergence, RenderTextureFormat.RFloat);

    FindKernels();

    // Initialize velocity
    if (kInitVelocity >= 0)
    {
      computeShader.SetVector("initialVelocity", initialVelocity);
      computeShader.SetTexture(kInitVelocity, "velocityWrite", velocity);
      DispatchFull(kInitVelocity);
    }

    // Build constant source (one-time sphere) if requested
    if (addConstantSource && kAddSource >= 0)
    {
      CreateSourceTexture();
      if (densitySource)
      {
        Graphics.CopyTexture(densitySource, densityA);
      }
      CacheSourceSettings();
    }

    UpdateBoundsFromTransform();
    PushStaticRenderParams();

    // Disable shadow casting for volumetric rendering
    if (volumeRenderer)
    {
      volumeRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
      volumeRenderer.receiveShadows = false;
    }
  }

  void Update()
  {
    float dt = Time.deltaTime;
    if (dt <= 0) return;

    // Set global simulation params
    computeShader.SetVector("gridSize", new Vector4(gridSize.x, gridSize.y, gridSize.z, 0));
    computeShader.SetFloat("deltaTime", dt);
    float safeCellSize = Mathf.Max(1e-4f, cellSize);
    computeShader.SetFloat("cellSize", safeCellSize);

    RebuildSourceTextureIfNeeded();


    //  source (add_source stage)
    if (kAddSource >= 0 && addConstantSource && densitySource)
    {
      float currentSourceScale = sourceScale;

      // Smooth cycling: inject strongly at start, fade out, then reset
      if (smartSourceInjection)
      {
        sourceTimer += dt;

        // Calculate smooth fade using cosine curve
        float cycleProgress = sourceTimer / sourceCycleTime;
        if (cycleProgress >= 1.0f)
        {
          sourceTimer = 0f;
          cycleProgress = 0f;
        }

        // Fade from 1.0 to 0.0 smoothly over the cycle
        float fadeAmount = Mathf.Cos(cycleProgress * Mathf.PI) * 0.5f + 0.5f;
        currentSourceScale *= fadeAmount;
      }

      computeShader.SetFloat("sourceScale", currentSourceScale);
      computeShader.SetTexture(kAddSource, "densityWrite", densityA);
      computeShader.SetTexture(kAddSource, "sourceDensity", densitySource);
      DispatchFull(kAddSource);
    }

    // Add velocity noise/source: write into velocityB then swap
    if (kAddVelocitySource >= 0)
    {
      computeShader.SetTexture(kAddVelocitySource, "velocityWrite", velocityB);
      computeShader.SetTexture(kAddVelocitySource, "velocityRead", velocity);
      // enviar posição/escala da fonte se necessário
      DispatchFull(kAddVelocitySource);
      Swap(ref velocity, ref velocityB);
    }
    if (projectVelocity)
    {
      ProjectVelocityField();
    }




    // 3. Diffusion (Jacobi) on density
    if (kDiffuse >= 0 && diffusionRate > 0)
    {
      float dx = safeCellSize;
      float alphaVal = (dx * dx) / (diffusionRate * dt);
      float rBetaVal = 1.0f / (6.0f + alphaVal);

      computeShader.SetFloat("alpha", alphaVal);
      computeShader.SetFloat("rBeta", rBetaVal);

      for (int i = 0; i < diffusionIterations; i++)
      {
        computeShader.SetTexture(kDiffuse, "bufferRead", densityA);
        computeShader.SetTexture(kDiffuse, "initialBuffer", densityA);
        computeShader.SetTexture(kDiffuse, "bufferWrite", densityB);
        DispatchFull(kDiffuse);
        Swap(ref densityA, ref densityB);
      }
    }

    // 4. Advection: densityA -> densityB
    if (kAdvect >= 0)
    {
      computeShader.SetTexture(kAdvect, "velocityRead", velocity);
      computeShader.SetTexture(kAdvect, "densityRead", densityA);
      computeShader.SetTexture(kAdvect, "densityWrite", densityB);
      DispatchFull(kAdvect);
      Swap(ref densityA, ref densityB);
    }

    // 5. Lifecycle: decay density over time
    if (kLifecycle >= 0 && decayRate > 0)
    {
      computeShader.SetFloat("decayRate", decayRate);
      computeShader.SetTexture(kLifecycle, "densityWrite", densityA);
      DispatchFull(kLifecycle);
    }

    Light sun = RenderSettings.sun;
    if (sun == null) sun = Object.FindAnyObjectByType<Light>(); // fallback
    if (sun != null)
    {
      Vector3 sunDir = -sun.transform.forward;
      rayMarchMaterial.SetVector("_LightDir", sunDir);
      rayMarchMaterial.SetColor("_LightCol", sun.color * sun.intensity);
      rayMarchMaterial.SetFloat("_gAnisotropy", gAnisotropy);
    }

    // 6. Update material
    if (rayMarchMaterial)
    {
      rayMarchMaterial.SetTexture("_DensityTex", densityA);
      rayMarchMaterial.SetVector("_GridSize", (Vector3)gridSize);
      rayMarchMaterial.SetVector("_BoundsMin", boundsMin);
      rayMarchMaterial.SetVector("_BoundsSize", boundsSize);
      rayMarchMaterial.SetColor("_CloudColor", cloudColor);
      rayMarchMaterial.SetColor("_DarkColor", darkCloudColor);
      rayMarchMaterial.SetFloat("_Absorption", absorption);
      rayMarchMaterial.SetFloat("_DensitySharpness", densitySharpness);
      rayMarchMaterial.SetFloat("_G_Anisotropy", gAnisotropy);
      rayMarchMaterial.SetInt("_Steps", Mathf.Max(4, rayMarchSteps));
      rayMarchMaterial.SetInt("_DebugBounds", debugBounds ? 1 : 0);
    }
  }

  void OnDestroy()
  {
    ReleaseRT(densityA);
    ReleaseRT(densityB);
    ReleaseRT(velocity);
    ReleaseRT(velocityB); // release velocityB
    ReleaseRT(pressureA);
    ReleaseRT(pressureB);
    ReleaseRT(divergence);
    ReleaseRT(densitySource);
  }

  // ------------------------------------------------------
  // Allocation / Setup
  // ------------------------------------------------------
  void Allocate3DTexture(ref RenderTexture rt, RenderTextureFormat fmt)
  {
    var desc = new RenderTextureDescriptor(
        gridSize.x,
        gridSize.y,
        fmt,
        0
        );
    desc.dimension         = TextureDimension.Tex3D;
    desc.volumeDepth       = gridSize.z;
    desc.enableRandomWrite = true;
    desc.msaaSamples       = 1;
    desc.mipCount          = 1;
    desc.autoGenerateMips  = false;
    desc.depthBufferBits   = 0;

    rt = new RenderTexture(desc);
    rt.wrapMode   = TextureWrapMode.Clamp;
    rt.filterMode = FilterMode.Bilinear;
    rt.enableRandomWrite = true;
    rt.Create();
  }

  void FindKernels()
  {
    kInitVelocity = SafeFind("InitVelocityKernel");
    kInject       = SafeFind("InjectKernel");
    kAddSource    = SafeFind("AddSourceKernel");
    kAddVelocitySource  = SafeFind("AddVelocitySourceKernel");
    kAdvect       = SafeFind("AdvectKernel");
    kDiffuse      = SafeFind("DiffuseKernel");
    kLifecycle    = SafeFind("LifecycleKernel");
    kDivergence   = SafeFind("DivergenceKernel");
    kPressureJacobi = SafeFind("PressureJacobiKernel");
    kSubtractGradient = SafeFind("SubtractGradientKernel");
  }

  int SafeFind(string kernel)
  {
    try
    {
      int k = computeShader.FindKernel(kernel);
      return k;
    }
    catch
    {
      Debug.LogWarning($"Kernel '{kernel}' not found in compute shader.", this);
      return -1;
    }
  }

  void UpdateBoundsFromTransform()
  {
    var t = volumeRenderer ? volumeRenderer.transform : transform;
    Vector3 center = t.position;
    Vector3 size   = t.localScale;
    boundsMin  = center - size * 0.5f;
    boundsSize = size;
  }

  void PushStaticRenderParams()
  {
    if (rayMarchMaterial)
    {
      rayMarchMaterial.SetVector("_BoundsMin", boundsMin);
      rayMarchMaterial.SetVector("_BoundsSize", boundsSize);
    }
  }

  void CacheSourceSettings()
  {
    cachedCloudCount = cloudCount;
    cachedCloudRadius = cloudRadius;
  }

  void RebuildSourceTextureIfNeeded()
  {
    if (!addConstantSource || kInject < 0)
      return;

    bool requiresRebuild = densitySource == null ||
      cloudCount != cachedCloudCount || 
      Mathf.Abs(cloudRadius - cachedCloudRadius) > 1e-4f;


    if (!requiresRebuild)
      return;

    CreateSourceTexture();

    if (densitySource)
    {
      Graphics.CopyTexture(densitySource, densityA);
    }

    CacheSourceSettings();
  }

  // One-time build of multiple cloud spheres using the InjectKernel
  void CreateSourceTexture()
  {
    if (densitySource != null)
    {
      densitySource.Release();
      densitySource = null;
    }

    var desc = new RenderTextureDescriptor(gridSize.x, gridSize.y, RenderTextureFormat.RFloat, 0)
    {
      dimension = TextureDimension.Tex3D,
      volumeDepth = gridSize.z,
      enableRandomWrite = true,
      msaaSamples = 1,
      mipCount = 1
    };
    densitySource = new RenderTexture(desc);
    densitySource.wrapMode = TextureWrapMode.Clamp;
    densitySource.filterMode = FilterMode.Point;
    densitySource.enableRandomWrite = true;
    densitySource.Create();

    

    if (kInject >= 0)
    {
      System.Random rand = new System.Random(42); // Fixed seed for consistency

      // Generate multiple cloud spheres at random positions
      for (int i = 0; i < cloudCount; i++)
      {
        float x = (float)rand.NextDouble() * gridSize.x;
        float y = (float)rand.NextDouble() * gridSize.y;
        float z = (float)rand.NextDouble() * gridSize.z; 
        // Vary the size slightly
        float sizeVariation = 0.5f + (float)rand.NextDouble() * 0.6f;
        float radius = cloudRadius * sizeVariation;

        // Vary the density slightly
        float densityVariation = 0.5f + (float)rand.NextDouble() * 0.5f; 

        computeShader.SetVector("injectPos", new Vector3(x, y, z));
        computeShader.SetFloat("injectRadius", radius);
        computeShader.SetFloat("injectValue", densityVariation);
        computeShader.SetTexture(kInject, "densityWrite", densitySource);
        DispatchFull(kInject);
      }
    }
  }

  // ------------------------------------------------------
  // Utility
  // ------------------------------------------------------
  void ProjectVelocityField()
  {
    if (kDivergence < 0 || kPressureJacobi < 0 || kSubtractGradient < 0)
      return;

    // Divergence computation (also clears pressureA)
    computeShader.SetTexture(kDivergence, "velocityRead", velocity);
    computeShader.SetTexture(kDivergence, "divergenceWrite", divergence);
    computeShader.SetTexture(kDivergence, "pressureWrite", pressureA);
    DispatchFull(kDivergence);

    RenderTexture pressureReadTex = pressureA;
    RenderTexture pressureWriteTex = pressureB;

    int iterations = Mathf.Max(0, pressureIterations);

    for (int i = 0; i < iterations; i++)
    {
      computeShader.SetTexture(kPressureJacobi, "pressureRead", pressureReadTex);
      computeShader.SetTexture(kPressureJacobi, "pressureWrite", pressureWriteTex);
      computeShader.SetTexture(kPressureJacobi, "divergenceRead", divergence);
      DispatchFull(kPressureJacobi);
      Swap(ref pressureReadTex, ref pressureWriteTex);
    }

    // IMPORTANT: read from current velocity, write into velocityB, then swap
    computeShader.SetTexture(kSubtractGradient, "pressureRead", pressureReadTex);
    computeShader.SetTexture(kSubtractGradient, "velocityRead", velocity);
    computeShader.SetTexture(kSubtractGradient, "velocityWrite", velocityB);
    DispatchFull(kSubtractGradient);
    Swap(ref velocity, ref velocityB);
  }

  void DispatchFull(int kernel)
  {
    if (kernel < 0) return;
    int gx = Mathf.CeilToInt(gridSize.x / 8.0f);
    int gy = Mathf.CeilToInt(gridSize.y / 8.0f);
    int gz = Mathf.CeilToInt(gridSize.z / 8.0f);
    computeShader.Dispatch(kernel, gx, gy, gz);
  }

  void Swap(ref RenderTexture a, ref RenderTexture b)
  {
    var tmp = a; a = b; b = tmp;
  }

  void ReleaseRT(RenderTexture rt)
  {
    if (rt != null)
    {
      rt.Release();
    }
  }
}
