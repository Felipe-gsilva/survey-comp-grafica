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

  public Vector3Int gridSize = new Vector3Int( 128, 96, 128);

  [Tooltip("Constant diffusion rate (small value ~0.02 - 0.1).")]
  public float diffusionRate = 0.03f;
  [Tooltip("Jacobi iterations for diffusion (more = smoother but slower).")]
  public int diffusionIterations = 18;
  [Tooltip("Blend factor between pure advection and diffusion (0 = pure advection, 1 = pure diffusion).")]
  [Range(0f, 1f)]
  public float diffusionBlend = 0.4f;

  [Tooltip("Density decay rate over time (0 = no decay, 1 = fast decay).")]
  [Range(0f, 1f)]
  public float decayRate = 0.35f;

  [Header("Pressure Projection")]
  public bool projectVelocity = true;
  [Tooltip("Jacobi iterations for pressure solve (divergence-free enforcement).")]
  [Range(0, 100)]
  public int pressureIterations = 40;
  [Tooltip("Grid cell size used for divergence/gradient (world units).")]
  public float cellSize = 1f;

  [Header("Velocity Init / Sources")]
  public Vector3 initialVelocity = new Vector3(0, 0.10f, 0);
  [Header("Wind")]
  [Tooltip("Direção e força globais do vento.")]
  public Vector3 windDirection = new Vector3(1.0f, 0.0f, 0.0f);
  [Tooltip("Multiplicador para a força do vento.")]
  [Range(0f, 10f)]
  public float windStrength = 0.3f;

  [Tooltip("Amortecimento da velocidade (0.99 = leve, 0.9 = forte).")]
  [Range(0.9f, 1.0f)]
  public float velocityDamping = 0.995f;


  [Header("Generic Source (Prebaked)")]
  public bool addConstantSource = true;
  public float sourceScale = 0.6f;


  [Tooltip("Time in seconds for source injection cycle (high = slower cycle)")]
  [Range(5f, 60f)]
  public float sourceCycleTime = 20f;

  private float sourceTimer = 0f;

  [Tooltip("Number of cloud spheres to generate")]
  [Range(1, 500)]
  public int cloudCount = 36;

  [Tooltip("Radius of each cloud sphere")]
  [Range(5f, 40f)]
  public float cloudRadius = 18f;

  [Header("Rendering")]
  public Material rayMarchMaterial;
  public MeshRenderer volumeRenderer;
  [Range(0.1f, 5.0f)]
  public float densitySharpness = 2.5f;
  [Range(4, 256)]
  public int rayMarchSteps = 96;
  public Color cloudColor = new Color(1.0f, 1.0f, 1.0f);
  public Color darkCloudColor = new Color(0.85f, 0.85f, 0.85f); // Mais claro para evitar bordas pretas

  [Range(0f, 2f)]
  public float absorption = 0.35f;

  public bool debugBounds = false;

  [Header("Auto Apply Recommended (Optional)")]
  [Tooltip("If enabled, overrides current component values & cube transform with recommended realistic cloud settings on Awake.")]
  public bool applyRecommendedDefaults = false;

  // Internal RenderTextures
  private RenderTexture densityA, densityB;
  private RenderTexture velocity, velocityB; 
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

    // Optionally override current inspector values for a consistent starting point
    if (applyRecommendedDefaults)
    {
      // Simulation tuning
      gridSize = new Vector3Int(128, 96, 128);
      diffusionRate = 0.025f;
      diffusionIterations = 20;
      diffusionBlend = 0.35f;
      decayRate = 0.30f;
      pressureIterations = 48;
      cellSize = 1f;
      initialVelocity = new Vector3(0f, 0.05f, 0f);
      windDirection = new Vector3(1f, 0f, 0f);
      windStrength = 0.25f;
      velocityDamping = 0.997f;
      addConstantSource = true;
      sourceScale = 0.8f;
      cloudCount = 60;
      cloudRadius = 22f;

      // Rendering tuning
      densitySharpness = .5f;
      rayMarchSteps = 128;
      cloudColor = new Color(1f, 1f, 1f);
      darkCloudColor = new Color(0.8f, 0.8f, 0.8f);
      absorption = .04f;
      debugBounds = false;

      // Recommended physical volume size (world units)
      if (volumeRenderer)
      {
        var t = volumeRenderer.transform;
        t.localScale = new Vector3(200f, 100f, 200f);
      }
    }

    Allocate3DTexture(ref densityA, RenderTextureFormat.RFloat, TextureWrapMode.Repeat);
    Allocate3DTexture(ref densityB, RenderTextureFormat.RFloat, TextureWrapMode.Repeat);
    
    Allocate3DTexture(ref velocity, RenderTextureFormat.ARGBFloat);
    Allocate3DTexture(ref velocityB, RenderTextureFormat.ARGBFloat); 
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

    // Build constant source if requested
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
    computeShader.SetVector("windDirection", windDirection.normalized);
    computeShader.SetFloat("windStrength", windStrength);
    computeShader.SetFloat("diffusionBlend", diffusionBlend);
    computeShader.SetFloat("velocityDamping", velocityDamping);

    RebuildSourceTextureIfNeeded();


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
    

    //  source (add_source stage)
    if (kAddSource >= 0 && addConstantSource && densitySource)
    {
      float currentSourceScale = sourceScale;

      computeShader.SetFloat("sourceScale", currentSourceScale);
      computeShader.SetTexture(kAddSource, "densityWrite", densityA);
      computeShader.SetTexture(kAddSource, "sourceDensity", densitySource);
      DispatchFull(kAddSource);
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
  void Allocate3DTexture(ref RenderTexture rt, RenderTextureFormat fmt, TextureWrapMode wrapMode = TextureWrapMode.Clamp)
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
    rt.wrapMode   = wrapMode;
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
    densitySource.wrapMode = TextureWrapMode.Repeat;
    densitySource.filterMode = FilterMode.Bilinear;    
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
