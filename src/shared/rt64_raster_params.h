//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RasterParams {
        uint renderIndex;
        // Which of its two depth bounds a decal's prebuilt pixel shader writes on WebGPU, 1 the high one and 0 the low
        // one (RasterPS.hlsl, DECAL_DEPTH_BOUNDS); nothing else reads it.
        uint decalDepthHigh;
        // Two words, not a uint2 after decalDepthHigh: WGSL starts a vector on 8 bytes, which would move what follows.
        uint padding1;
        uint padding2;
        float2 screenScale;
        float2 screenOffset;
    };
#ifdef HLSL_CPU
    static_assert(sizeof(RasterParams) == 32, "RasterParams must keep the layout WGSL gives it.");
#endif
#ifdef HLSL_CPU
};
#endif