//
// RT64
//

#pragma once

#include "rt64_sampler_library.h"

namespace RT64 {
    struct ShaderRecord {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
    };

    struct ShaderLibrary {
        // The swap chain's format. A browser's canvas takes rgba8unorm or bgra8unorm, and Firefox (157 on Windows 10)
        // keeps every frame a bgra8unorm canvas presented in video memory, about 13 MB a frame at 1080p, until the GPU
        // runs out and the frame rate collapses. With rgba8unorm it frees them.
#if defined(__EMSCRIPTEN__)
        static constexpr RenderFormat SwapChainFormat = RenderFormat::R8G8B8A8_UNORM;
#else
        static constexpr RenderFormat SwapChainFormat = RenderFormat::B8G8R8A8_UNORM;
#endif

        SamplerLibrary samplerLibrary;
        bool usesHDR = false;
        bool usesHardwareResolve = false;

        // All shaders.
        ShaderRecord bicubicScaling;
        ShaderRecord boxFilter;
        ShaderRecord compose;
        ShaderRecord debug;
        ShaderRecord fbChangesClear;
        ShaderRecord fbChangesDrawColor;
        ShaderRecord fbChangesDrawDepth;
        ShaderRecord fbReadAnyChanges;
        ShaderRecord fbReadAnyFull;
        ShaderRecord fbReinterpret;
        ShaderRecord fbWriteColor;
        ShaderRecord fbWriteDepth;
        ShaderRecord fbWriteDepthMS;
        ShaderRecord gaussianFilterRGB3x3;
        ShaderRecord histogramAverage;
        ShaderRecord histogramClear;
        ShaderRecord histogramSet;
        ShaderRecord idle;
        ShaderRecord im3dLine;
        ShaderRecord im3dPoint;
        ShaderRecord im3dTriangle;
        ShaderRecord luminanceHistogram;
        ShaderRecord postProcess;
        ShaderRecord rspModify;
        ShaderRecord rspProcess;
        ShaderRecord rspSmoothNormal;
        ShaderRecord rspVertexTestZ;
        ShaderRecord rspVertexTestZMS;
        ShaderRecord rspWorld;
        ShaderRecord rtCopyColorToDepth;
        ShaderRecord rtCopyDepthToColor;
        ShaderRecord rtCopyColorToDepthMS;
        ShaderRecord rtCopyDepthToColorMS;
        ShaderRecord textureDecode;
        ShaderRecord textureCopy;
        ShaderRecord textureResolve;
        ShaderRecord videoInterfaceLinear;
        ShaderRecord videoInterfaceNearest;
        ShaderRecord videoInterfacePixel;

        ShaderLibrary(bool usesHDR, bool usesHardwareResolve);
        ~ShaderLibrary();
        void setupCommonShaders(RenderInterface *rhi, RenderDevice *device);
        void setupMultisamplingShaders(RenderInterface *rhi, RenderDevice *device, const RenderMultisampling &multisampling);
    };
};
