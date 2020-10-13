#include <metal_stdlib>
using namespace metal;

float noise(float2 coord, float2 multiplierA, float2 multiplierB, float2 multiplierC, float3 timeMultipliers, float time) {
    return 0.333 * (sin(dot(multiplierA, coord) + timeMultipliers.x * time) +
                    sin(dot(multiplierB, coord) + timeMultipliers.y * time) +
                    sin(dot(multiplierC, coord) + timeMultipliers.z * time));
}

kernel void khyberPass(texture2d<float, access::write> o[[texture(0)]],
                       constant float &time [[buffer(0)]],
                       constant float2 *touchEvent [[buffer(1)]],
                       constant int &numberOfTouches [[buffer(2)]],
                       ushort2 gid [[thread_position_in_grid]]) {

  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 p = float2(gid.xy);
  float2 uv = p.xy / res.y;
  float2 uvOffset;

  uvOffset.x = 0.2 * sin(time * 0.41 + 0.7) * pow(abs(uv.y - 0.5), 3.1) - sin(time * 0.07 + 0.1);
  uvOffset.y = -time * 0.03 + 0.05 * sin(time * 0.3) * pow(abs(uv.x - 0.5), 1.8);
  uv += uvOffset;

  const float cellResolution = 4.0;
  const float lineSmoothingWidth = 0.15;
  float2 localUV = fract(uv * cellResolution);
  float2 cellCoord = floor(uv * cellResolution);

  float2 angle = 5 * normalize(float2(noise(cellCoord, float2(1.7, 0.9), float2(2.6, 1.1), float2(0.0), float3(0.55, 0.93, 0.0), time),
                                        noise(cellCoord, float2(0.6, 1.9), float2(1.3, 0.3), float2(0.0), float3(1.25, 0.83, 0.0), time)));

  float v = smoothstep(-lineSmoothingWidth, lineSmoothingWidth, abs(fract(dot(localUV, angle) + 3.6 * time) - 0.5) - 0.25);

  const float borderSmoothingWidth = 0.02;
  float2 centeredLocalUV = localUV - float2(0.5);
  const float borderDistance = 0.75;
  v = max(v, max(smoothstep(-borderSmoothingWidth, borderSmoothingWidth, abs(centeredLocalUV.x) - borderDistance), smoothstep(-borderSmoothingWidth, borderSmoothingWidth, abs(centeredLocalUV.y) - borderDistance)));

  float4 color = v == 1 ? float4(1) : float4(0.004, 0.004, 0.137, 1.0);
  o.write(color, gid);
}
