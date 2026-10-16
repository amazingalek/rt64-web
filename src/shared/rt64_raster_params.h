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
        // Three words, not a uint3: WGSL starts a vec3 on 16 bytes, which would move what follows it.
        uint padding0;
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