//
// RT64
//

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <list>
#include <string>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "common/rt64_elapsed_timer.h"
#include "rt64_raster_shader.h"

namespace RT64 {
    struct RasterShaderCache {
        struct CompilationThread {
            RasterShaderCache *shaderCache;
            std::unique_ptr<std::thread> thread;
            std::atomic<bool> threadRunning;

            CompilationThread(RasterShaderCache *shaderCache);
            ~CompilationThread();
            void loop();
        };

        RenderDevice *device;
        std::unique_ptr<RasterShaderUber> shaderUber;
        OptimizerCacheSPIRV optimizerCacheSPIRV;
        ShaderCacheWGSL shaderCacheWGSL;
        std::mutex submissionMutex;
        std::queue<ShaderDescription> descQueue;
        std::mutex descQueueMutex;
        int32_t descQueueActiveCount = 0;
        std::condition_variable descQueueChanged;
        std::unordered_map<uint64_t, bool> shaderHashes;
        std::unordered_map<uint64_t, std::unique_ptr<RasterShader>> GPUShaders;
        std::mutex GPUShadersMutex;
        std::list<std::unique_ptr<CompilationThread>> compilationThreads;
        uint32_t threadCount;
        uint32_t ubershaderThreadCount;
        RenderShaderFormat shaderFormat;
        std::unique_ptr<ShaderCompiler> shaderCompiler;
        RenderMultisampling multisampling;
        bool usesHDR = false;
        bool specializedShaders = true;
        bool recordShaders = false;

        // The shaders specialized when the program was built, built before the game gets past its first frame.
        std::list<std::thread> prebuildThreads;
        std::mutex prebuildMutex;
        std::condition_variable prebuildChanged;
        uint32_t prebuildCount = 0;
        uint32_t prebuildDone = 0;
        uint32_t prebuildNext = 0;
        std::atomic<bool> prebuildDrawn = false;
        std::string prebuildText;
        ElapsedTimer prebuildTimer;

        RasterShaderCache(uint32_t threadCount, uint32_t ubershaderThreadCount);
        ~RasterShaderCache();
        void setup(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary, const RenderMultisampling &multisampling);
        void submit(const ShaderDescription &desc);
        void waitForAll();
        void startPrebuild();
        void prebuildLoop();
        void reportPrebuild(uint32_t done);
        void waitForPrebuild();
        void destroyAll();
        RasterShader *getGPUShader(const ShaderDescription &desc);
        RasterShaderUber *getGPUShaderUber() const;
        uint32_t shaderCount();
    };
};