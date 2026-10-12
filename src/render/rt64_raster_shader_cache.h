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
#include <vector>

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
        bool prebuiltShaders = false;
        bool recordShaders = false;

        // The shaders specialized when the program was built: the game waits past its first frame for the first
        // prebuildWaitCount, and the rest are built behind it, any a draw asks for first (prebuildUrgent).
        std::list<std::thread> prebuildThreads;
        std::mutex prebuildMutex;
        std::condition_variable prebuildChanged;
        uint32_t prebuildCount = 0;
        uint32_t prebuildDone = 0;
        uint32_t prebuildNext = 0;
        uint32_t prebuildWaitCount = 0;
        uint32_t prebuildWaitDone = 0;
        enum : uint8_t { PrebuildQueued, PrebuildHurried, PrebuildTaken };
        std::vector<uint8_t> prebuildState;
        std::queue<uint32_t> prebuildUrgent;
        std::unordered_map<uint64_t, uint32_t> prebuildIndices;
        std::atomic<bool> prebuildBusy = false;
        std::atomic<bool> prebuildDrawn = false;
        std::string prebuildText;
        ElapsedTimer prebuildTimer;

        RasterShaderCache(uint32_t threadCount, uint32_t ubershaderThreadCount);
        ~RasterShaderCache();
        void setup(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary, const RenderMultisampling &multisampling);
        void submit(const ShaderDescription &desc);
        void waitForAll();
        void startPrebuild();
        void prebuildLoop(uint32_t thread);
        void hurryPrebuild(uint64_t shaderHash);
        void reportPrebuild(uint32_t done);
        void waitForPrebuild();
        void destroyAll();
        RasterShader *getGPUShader(const ShaderDescription &desc);
        RasterShaderUber *getGPUShaderUber() const;
        uint32_t shaderCount();
    };
};