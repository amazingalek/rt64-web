//
// RT64
//

#include "shared/rt64_rsp_fog.h"
#include "shared/rt64_rsp_light.h"
#include "shared/rt64_rsp_lookat.h"
#include "shared/rt64_rsp_viewport.h"

#include "TextureGen.hlsli"
#include "WebGPU.hlsli"

#define GROUP_SIZE 64

struct RSPProcessCB {
    uint vertexStart;
    uint vertexCount;
    float prevFrameWeight;
    float curFrameWeight;
};

[[vk::push_constant]] ConstantBuffer<RSPProcessCB> gConstants : register(b0);
TYPED_BUFFER(float) srcPos : register(t1);
TYPED_BUFFER(float) srcVel : register(t2);
TYPED_BUFFER(float) srcTc : register(t3);
TYPED_BUFFER(float) srcTcVel : register(t4);
TYPED_BUFFER(uint) srcCol : register(t5);
TYPED_BUFFER(int) srcNorm : register(t6);
TYPED_BUFFER(uint) srcViewProjIndices : register(t7);
TYPED_BUFFER(uint) srcWorldIndices : register(t8);
TYPED_BUFFER(uint) srcFogIndices : register(t9);
TYPED_BUFFER(uint) srcLightIndices : register(t10);
TYPED_BUFFER(uint) srcLightCounts : register(t11);
TYPED_BUFFER(uint) srcLookAtIndices : register(t12);
StructuredBuffer<RSPViewport> rspViewportVector : register(t13);
StructuredBuffer<RSPFog> rspFogVector : register(t14);
StructuredBuffer<RSPLight> rspLightVector : register(t15);
StructuredBuffer<RSPLookAt> rspLookAtVector : register(t16);
StructuredBuffer<float4x4> viewProjTransforms : register(t17);
StructuredBuffer<float4x4> worldTransforms : register(t18);
RWStructuredBuffer<float4> dstPos : register(u19);
RWStructuredBuffer<float2> dstTc : register(u20);
RWStructuredBuffer<float4> dstCol : register(u21);

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint vertexIndex : SV_DispatchThreadID) {
    if (vertexIndex >= gConstants.vertexCount) {
        return;
    }

    const uint vertexOffsetIndex = gConstants.vertexStart + vertexIndex;
    const uint viewProjIndex = loadUint16(srcViewProjIndices, vertexOffsetIndex);
    const uint transformIndex = loadUint16(srcWorldIndices, vertexOffsetIndex);
    const float4x4 viewProjMat = viewProjTransforms[viewProjIndex];
    const float4x4 worldMat = worldTransforms[transformIndex];
    const uint posIndex = vertexOffsetIndex * 3;
    const uint normColIndex = vertexOffsetIndex * 4;
    const float3 norm = float3(
        loadInt8(srcNorm, normColIndex + 0) / 127.0f,
        loadInt8(srcNorm, normColIndex + 1) / 127.0f,
        loadInt8(srcNorm, normColIndex + 2) / 127.0f
    );

    // NDC Position.
    const float3 pos = float3(loadFloat(srcPos, posIndex + 0), loadFloat(srcPos, posIndex + 1), loadFloat(srcPos, posIndex + 2));
    const float3 vel = float3(loadFloat(srcVel, posIndex + 0), loadFloat(srcVel, posIndex + 1), loadFloat(srcVel, posIndex + 2));
    float4 tfPos = mul(mul(viewProjMat, worldMat), float4(pos - vel * (1.0f - gConstants.curFrameWeight), 1.0f));

    // Fog.
    const uint fogIndex = loadUint16(srcFogIndices, vertexOffsetIndex);
    float4 vertexColor;
    if (fogIndex > 0) {
        const RSPFog rspFog = rspFogVector[fogIndex - 1];
        const float fogAlpha = ((max(tfPos.z, 0) / tfPos.w) * rspFog.mul + rspFog.offset);
        vertexColor.a = clamp(fogAlpha / 255.0f, 0.0f, 1.0f);
    }
    else {
        vertexColor.a = loadUint8(srcCol, normColIndex + 3) / 255.0f;
    }

    // Texgen.
    const uint tcIndex = vertexOffsetIndex * 2;
    const uint lookAtIndex = loadUint16(srcLookAtIndices, vertexOffsetIndex);
    float2 tc = float2(loadFloat(srcTc, tcIndex + 0), loadFloat(srcTc, tcIndex + 1));
    if (lookAtIndex & RSP_LOOKAT_INDEX_ENABLED) {
        const bool textureGenLinear = lookAtIndex & RSP_LOOKAT_INDEX_LINEAR;
        const uint extractedIndex = (lookAtIndex >> RSP_LOOKAT_INDEX_SHIFT);
        const RSPLookAt lookAt = rspLookAtVector[extractedIndex];
        tc = computeTextureGen(tc, norm, lookAt, textureGenLinear, worldMat);
    }
    else {
        float2 tcVel = float2(loadFloat(srcTcVel, tcIndex + 0), loadFloat(srcTcVel, tcIndex + 1));
        tc -= tcVel * (1.0f - gConstants.curFrameWeight);
    }

    // Lighting.
    const uint lightIndex = loadUint16(srcLightIndices, vertexOffsetIndex);
    const uint lightCount = loadUint8(srcLightCounts, vertexOffsetIndex);
    if (lightCount > 0) {
        const uint ambientIndex = lightIndex + lightCount - 1;
        float3 resultColor = rspLightVector[ambientIndex].col;
        for (uint i = lightIndex; i < ambientIndex; i++) {
            const RSPLight light = rspLightVector[i];
            if (light.kc > 0) {
                resultColor += computePosLight(pos, norm, light, worldMat);
            }
            else {
                resultColor += computeDirLight(norm, light, worldMat);
            }
        }

        vertexColor.rgb = min(resultColor, 1.0f);
    }
    else {
        vertexColor.rgb = float3(
            loadUint8(srcCol, normColIndex + 0) / 255.0f, 
            loadUint8(srcCol, normColIndex + 1) / 255.0f, 
            loadUint8(srcCol, normColIndex + 2) / 255.0f
        );
    }
    
    // HACK: For handling geometry exactly at the near clip plane of the viewport. This hack can
    // probably be removed once the behavior of the RSP has been reviewed in this case.
    if (tfPos.w == 0.0f) {
        tfPos.w = 1e-6f;
    }

    // Convert to N64 screen position.
    const RSPViewport rspViewport = rspViewportVector[viewProjIndex];
    const float3 ndcPos = tfPos.xyz / float3(tfPos.w, -tfPos.w, tfPos.w);
    const float4 screenPos = float4(ndcPos * rspViewport.scale + rspViewport.translate, tfPos.w);
    dstPos[vertexOffsetIndex] = screenPos;
    dstTc[vertexOffsetIndex] = tc;
    dstCol[vertexOffsetIndex] = vertexColor;
}