Shader "Custom/RayMarchShader"
{
    Properties
    {
        _DensityTex ("Density Texture", 3D) = "" {}
        _CloudColor ("Cloud Color", Color) = (1, 1, 1, 1)
        _DarkColor ("Ambient/Shadow Color", Color) = (0.5, 0.5, 0.5, 1)
        _Absorption ("Absorption", Range(0, 2)) = 0.5
        _Steps ("Ray March Steps", Int) = 64
        
        _DensitySharpness ("Density Sharpness", Range(0.1, 5.0)) = 1.0 // Controlled by C#
        _G_Anisotropy ("Anisotropy (g)", Range(-0.9, 0.9)) = 0.4     // Controlled by C#
        
        _DebugBounds ("Debug Bounds", Int) = 0

        _LightDir ("Main Light Direction", Vector) = (0, -1, 0, 0)
        _LightCol ("Main Light Color", Color) = (1, 1, 1, 1)
    }
    SubShader
    {
        Tags 
        { 
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
        }
        
        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off
        ZWrite Off

        Pass
        {
            Name "VolumetricRayMarch"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // Texture and Sampler
            TEXTURE3D(_DensityTex);
            SAMPLER(sampler_DensityTex);

            // Properties
            CBUFFER_START(UnityPerMaterial)
                float3 _GridSize;
                float3 _BoundsMin;
                float3 _BoundsSize;
                float4 _CloudColor;
                float4 _DarkColor;
                float  _Absorption;
                int    _Steps;
                float  _DensitySharpness; // Now a uniform
                float  _G_Anisotropy;     // Now a uniform
                int    _DebugBounds;
                float4 _LightDir;
                float4 _LightCol;
            CBUFFER_END
            
            // Shared density threshold
            static const float densityThreshold = 0.05;

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float fogFactor : TEXCOORD1;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                
                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = vertexInput.positionCS;
                output.positionWS = vertexInput.positionWS;
                output.fogFactor = ComputeFogFactor(output.positionCS.z);
                
                return output;
            }
            
            // p = position in local volume space (0.0 to 1.0)
            float get_boundary_fade(float3 p)
            {
                float3 edge_dist = min(p, 1.0 - p);
                float fade_width = 0.2; 
                float fade_x = smoothstep(0.0, fade_width, edge_dist.x);
                float fade_y = smoothstep(0.0, fade_width, edge_dist.y);
                float fade_z = smoothstep(0.0, fade_width, edge_dist.z);
                return fade_x * fade_y * fade_z;
            }

            bool RayBox(float3 ro, float3 rd, float3 bmin, float3 bmax, out float tmin, out float tmax)
            {
                float3 inv = 1.0 / (rd + 1e-6);
                float3 t0 = (bmin - ro) * inv;
                float3 t1 = (bmax - ro) * inv;
                float3 tsmaller = min(t0, t1);
                float3 tbigger  = max(t0, t1);
                tmin = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
                tmax = min(tbigger.x,  min(tbigger.y,  tbigger.z));
                return tmax > max(tmin, 0.0);
            }
            
            // Remap density function (now uses uniform)
            float remap_density(float rawDensity)
            {
                // Use the uniform _DensitySharpness passed from C#
                return pow(saturate((rawDensity - densityThreshold) / (1.0 - densityThreshold)), _DensitySharpness);
            }

            // Schlick phase function for light scattering
            // g = anisotropy (-1 = back, 0 = uniform, 1 = forward)
            float SchlickPhase(float cosTheta, float g)
            {
                float k = 1.55 * g - 0.55 * (g * g * g);
                float kCosTheta = k * cosTheta;
                // Avoid division by zero
                float denom = (1.0 - kCosTheta) * (1.0 - kCosTheta);
                return (1.0 - k * k) / (4.0 * PI * max(0.001, denom));
            }

            // Shadow raymarch
            // Casts a ray from worldPos towards lightDir to find shadow
            float SampleShadow(float3 worldPos, float3 lightDir)
            {
                int shadowSteps = 16; // Fewer steps for performance
                float shadowStepSize = 0.5; 
                float transmittance = 1.0;
                
                [loop]
                for (int i = 1; i <= shadowSteps; i++)
                {
                    float3 shadowSamplePos = worldPos + lightDir * shadowStepSize * (float)i;
                    float3 uvw = (shadowSamplePos - _BoundsMin) / _BoundsSize;
                    
                    // If outside box, it's lit (no shadow)
                    if (uvw.x < 0 || uvw.x > 1 || uvw.y < 0 || uvw.y > 1 || uvw.z < 0 || uvw.z > 1)
                    {
                        continue; 
                    }

                    float rawDensity = SAMPLE_TEXTURE3D_LOD(_DensityTex, sampler_DensityTex, uvw, 0).r;
                    float shadowDensity = remap_density(rawDensity);
                    
                    // Apply absorption
                    float absorb = exp(-shadowDensity * _Absorption * shadowStepSize);
                    transmittance *= absorb;
                    
                    if (transmittance < 0.01)
                        return 0.0;
                }
                return transmittance;
            }


            half4 frag(Varyings input) : SV_Target
            {
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(input.positionWS - ro);
                
                // Get main light (sun)
                float3 lightDir = _LightDir;
                float3 lightColor = _LightCol.rgb;
                float t0, t1;
                if (!RayBox(ro, rd, _BoundsMin, _BoundsMin + _BoundsSize, t0, t1))
                {
                    discard;
                }

                t0 = max(t0, 0.0);
                float dist = t1 - t0;

                int steps = max(4, _Steps);
                float stepSize = dist / (float)steps;
                float3 startPos = ro + rd * t0;

                float3 accum = 0.0;
                float transmittance = 1.0;

                [loop]
                for (int s = 0; s < steps; s++)
                {
                    float3 p = startPos + rd * ((float)s * stepSize + stepSize * 0.5);
                    float3 uvw = (p - _BoundsMin) / _BoundsSize;
                    uvw = saturate(uvw);

                    float rawDensity = SAMPLE_TEXTURE3D_LOD(_DensityTex, sampler_DensityTex, uvw, 0).r;
                    float density = remap_density(rawDensity);
                    density *= get_boundary_fade(uvw);

                    if (density > 0.01)
                    {
                        // --- NEW LIGHTING LOGIC ---
                        
                        // 1. Calculate self-shadowing (0=shadow, 1=lit)
                        float shadow = SampleShadow(p, lightDir);
                        
                        // 2. Calculate phase function
                        float cosTheta = dot(-rd, lightDir);
                        float phase = SchlickPhase(cosTheta, _G_Anisotropy);

                        // 3. Combine main light, shadow, and phase
                        float3 scattering = _CloudColor.rgb * lightColor * phase * shadow;
                        
                        // 4. Add ambient light (using _DarkColor)
                        float3 ambient = _DarkColor.rgb; 
                        float3 litColor = scattering + ambient;
                        
                        float absorb = exp(-density * _Absorption * stepSize);
                        
                        accum += transmittance * (1.0 - absorb) * litColor;
                        transmittance *= absorb;
                        
                        if (transmittance < 0.01)
                            break;
                    }
                }

                float alpha = 1.0 - transmittance;

                if (_DebugBounds == 1)
                {
                    float3 rel = (input.positionWS - (_BoundsMin + 0.5 * _BoundsSize)) / (_BoundsSize * 0.5);
                    float edge = step(0.95, max(abs(rel.x), max(abs(rel.y), abs(rel.z))));
                    accum = lerp(accum, float3(1, 0, 0), edge);
                    alpha = max(alpha, edge * 0.3);
                }

                accum = MixFog(accum, input.fogFactor);
                alpha = saturate(alpha);

                if (alpha < 0.01)
                    discard;

                return half4(accum, alpha);
            }
            ENDHLSL
        }
    }
    FallBack Off
}

