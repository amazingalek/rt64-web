//
// RT64
//

#pragma once

#include <stdint.h>

// The shaders a game prebuilds on WebGPU (RT64_PREBUILT_SHADER_LIST in CMakeLists.txt, RasterShaderCache::startPrebuild()):
// a shader description each, and where its vertex and pixel shaders' WGSL is in the text that RT64PrebuiltShaderData
// decompresses to (zstd), zero-terminated, the sizes counting the zero: the pixel shader with dual-source blending, and
// the one for a device without it. shader_specializer writes them.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t otherModeL;
    uint32_t otherModeH;
    uint32_t combinerL;
    uint32_t combinerH;
    uint32_t flags;
    uint32_t vertexOffset;
    uint32_t vertexSize;
    uint32_t pixelOffset;
    uint32_t pixelSize;
    uint32_t pixelSingleOffset;
    uint32_t pixelSingleSize;
} RT64PrebuiltShader;

extern const RT64PrebuiltShader RT64PrebuiltShaderList[];
extern const uint32_t RT64PrebuiltShaderCount;
extern const uint32_t RT64PrebuiltShaderTextSize;
extern const uint32_t RT64PrebuiltShaderDataSize;
extern const unsigned char RT64PrebuiltShaderData[];

#ifdef __cplusplus
}
#endif
