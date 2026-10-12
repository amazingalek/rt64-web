//
// RT64
//

#include "rt64_raster_shader_cache.h"

#include "common/rt64_thread.h"

#if defined(__EMSCRIPTEN__)
#   include <emscripten/emscripten.h>
#endif

#if defined(RT64_PREBUILT_SHADERS)
#   include <cstring>
#   include <zstd.h>
#   include "common/rt64_elapsed_timer.h"
#   include "rt64_prebuilt_shaders.h"
#endif

// On WebGPU a specialized RasterPS takes 1.4-2 s to build in the browser's GPU process (Chrome, RTX 2070). plume's
// WebGPU backend builds it with createRenderPipelineAsync and the compilation thread waits for it, so the ubershader
// draws meanwhile, as everywhere else.
#define ENABLE_OPTIMIZED_SHADER_GENERATION

namespace RT64 {
    // RasterShaderCache::CompilationThread
    
    RasterShaderCache::CompilationThread::CompilationThread(RasterShaderCache *shaderCache) {
        assert(shaderCache != nullptr);

        this->shaderCache = shaderCache;

        thread = std::make_unique<std::thread>(&CompilationThread::loop, this);
        threadRunning = false;
    }

    RasterShaderCache::CompilationThread::~CompilationThread() {
        threadRunning = false;
        shaderCache->descQueueChanged.notify_all();
        thread->join();
        thread.reset(nullptr);
    }

    void RasterShaderCache::CompilationThread::loop() {
        Thread::setCurrentThreadName("RT64 Shader");

        // The shader compilation thread should have idle priority by default as the application can use the ubershader in the meantime.
        Thread::setCurrentThreadPriority(Thread::Priority::Idle);

        threadRunning = true;

        while (threadRunning) {
            ShaderDescription shaderDesc;
            bool fromPriorityQueue = false;
            
            // Check the top of the queue or wait if it's empty.
            {
                std::unique_lock<std::mutex> queueLock(shaderCache->descQueueMutex);
                shaderCache->descQueueActiveCount--;
                shaderCache->descQueueChanged.wait(queueLock, [this]() {
                    return !threadRunning || !shaderCache->descQueue.empty();
                });

                shaderCache->descQueueActiveCount++;
                if (!shaderCache->descQueue.empty()) {
                    shaderDesc = shaderCache->descQueue.front();
                    shaderCache->descQueue.pop();
                    fromPriorityQueue = true;
                }
            }
            
            // Compile the shader at the top of the queue.
            if (fromPriorityQueue) {
                assert((shaderCache->shaderUber != nullptr) && "Ubershader should've been created by the time a new shader is submitted to the cache.");
                const RenderPipelineLayout *uberPipelineLayout = shaderCache->shaderUber->pipelineLayout.get();
                const RenderMultisampling multisampling = shaderCache->multisampling;
                std::unique_ptr<RasterShader> newShader = std::make_unique<RasterShader>(shaderCache->device, shaderDesc, uberPipelineLayout, shaderCache->shaderFormat, multisampling, shaderCache->shaderCompiler.get(), &shaderCache->optimizerCacheSPIRV, &shaderCache->shaderCacheWGSL);

                {
                    const std::unique_lock<std::mutex> lock(shaderCache->GPUShadersMutex);
                    shaderCache->GPUShaders[shaderDesc.hash()] = std::move(newShader);
                }
            }
        }
    }

    // RasterShaderCache

    RasterShaderCache::RasterShaderCache(uint32_t threadCount, uint32_t ubershaderThreadCount) {
        assert(threadCount > 0);

        this->threadCount = threadCount;
        this->ubershaderThreadCount = ubershaderThreadCount;

#ifdef ENABLE_OPTIMIZED_SHADER_GENERATION
#   ifdef _WIN32
        shaderCompiler = std::make_unique<ShaderCompiler>();
#   endif

        descQueueActiveCount = threadCount;

        for (uint32_t t = 0; t < threadCount; t++) {
            compilationThreads.push_back(std::make_unique<CompilationThread>(this));
        }
#endif
    }

    RasterShaderCache::~RasterShaderCache() {
        compilationThreads.clear();
        for (std::thread &thread : prebuildThreads) {
            thread.join();
        }
    }

    void RasterShaderCache::setup(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary, const RenderMultisampling &multisampling) {
        assert(device != nullptr);

        this->device = device;
        this->shaderFormat = shaderFormat;
        this->multisampling = multisampling;

        shaderUber = std::make_unique<RasterShaderUber>(device, shaderFormat, multisampling, shaderLibrary, ubershaderThreadCount);
        usesHDR = shaderLibrary->usesHDR;

        // Where building a pipeline holds up the device (WebGPU in Firefox), a specialized shader stalls the frames
        // for longer than the ubershader it replaces costs them, and a race asks for a new one every few seconds.
        // What isn't prebuilt draws with the ubershader there.
        specializedShaders = device->getCapabilities().backgroundPipelineBuilds;

        // The prebuilt shaders are built wherever a game has a list, whatever the browser and the adapter: they are
        // built before the game starts, when holding up the device costs nothing, and without them a weak GPU draws
        // everything with the ubershader, a frame a second (startPrebuild()).
#   if defined(RT64_PREBUILT_SHADERS)
        prebuiltShaders = (shaderFormat == RenderShaderFormat::WGSL) && (multisampling.sampleCount == 1);
#   endif

#   if defined(__EMSCRIPTEN__)
        // A page opened with ?record-shaders prints every shader the game asks for, for a list of prebuilt shaders
        // (RT64_PREBUILT_SHADER_LIST in CMakeLists.txt says how it is used).
        recordShaders = MAIN_THREAD_EM_ASM_INT({ return new URLSearchParams(location.search).has('record-shaders') ? 1 : 0; }) != 0;
#   endif

        // Initialize the re-spirv optimizer cache.
        if (shaderFormat == RenderShaderFormat::SPIRV) {
            optimizerCacheSPIRV.initialize();
        }

        // Create the spec constant shaders before anything draws. plume's WebGPU backend learns from each new shader which
        // stages read which bindings, and when that changes every pipeline gets a new layout: the specialized ones that
        // were drawn with would then be built again in place, holding up the browser's GPU process for seconds. Without
        // specialized or prebuilt shaders nothing needs them, and each is megabytes of WGSL for the browser to parse.
        if ((shaderFormat == RenderShaderFormat::WGSL) && (specializedShaders || prebuiltShaders)) {
            const RenderShader *vertexShader = nullptr;
            const RenderShader *pixelShader = nullptr;
            for (bool smoothShade : { false, true }) {
                shaderCacheWGSL.get(device, smoothShade, multisampling.sampleCount > 1, vertexShader, pixelShader);
            }
        }

        startPrebuild();
    }

    // The prebuilt shaders are the ones a game recorded (?record-shaders), specialized when the program was built and
    // built here before the game gets past its first frame, so a weak GPU never draws a frame with the ubershader, which
    // costs several times what they do per pixel (on a laptop's Intel graphics it was the difference between a frame
    // a second and 60), and no shader is built in the middle of a race. Built from the ubershader's WGSL with the spec
    // constants as overrides, a specialized RasterPS took 6 s cold in Chrome (RTX 2070); specialized beforehand, 1-1.6 s,
    // and one game's 105 took 13-18 s together. Chrome keeps them for no later visit: its pipeline cache is far smaller.
    // A pipeline is built for the WebGPU layout a draw with its pipeline layout resolved (plume_webgpu.h), so they all
    // wait for the first frame's draws. waitForPrebuild() then holds the next frame until the first part of the list is
    // done, the shaders the game's first screens use (above the list's "background" line; all of them without one), so
    // a long list doesn't keep the player waiting: GoldenEye's 340 took ~85 s. Building that part takes the browser's
    // GPU process a few at once: there is nothing else for it to do meanwhile. Each one of it done is reported to the
    // page as Module.rt64ShaderProgress(done, count), for it to show. Then one thread builds the rest behind the game,
    // in the list's order but first any a draw asks for (hurryPrebuild()), which draws with the ubershader until its
    // shader is built; more at once take cores the game's frames need. They are made without MSAA, as the web draws,
    // with RasterPS both with dual-source blending and without, for a device that lacks it. Where pipelines are built
    // in line with the device's other work (Firefox) a build behind the game would stall its frames, so the game waits
    // for the whole list there, and only the shaders the list missed are left to the ubershader (setup()). A page opened
    // with ?record-shaders prebuilds none, so that it prints every shader the game asks for, in the order it asks.
    void RasterShaderCache::startPrebuild() {
#   if defined(RT64_PREBUILT_SHADERS)
        if (!prebuiltShaders || recordShaders) {
            return;
        }

        prebuildText.resize(RT64PrebuiltShaderTextSize);
        const size_t textSize = ZSTD_decompress(prebuildText.data(), prebuildText.size(), RT64PrebuiltShaderData, RT64PrebuiltShaderDataSize);
        if (textSize != RT64PrebuiltShaderTextSize) {
            fprintf(stderr, "RT64: the prebuilt shaders didn't decompress: %s\n", ZSTD_isError(textSize) ? ZSTD_getErrorName(textSize) : "wrong size");
            return;
        }

        // What is prebuilt is never queued for the compilation threads.
        {
            std::unique_lock<std::mutex> queueLock(submissionMutex);
            for (uint32_t i = 0; i < RT64PrebuiltShaderCount; i++) {
                const RT64PrebuiltShader &entry = RT64PrebuiltShaderList[i];
                ShaderDescription desc;
                memset(&desc, 0, sizeof(desc));
                desc.otherMode.L = entry.otherModeL;
                desc.otherMode.H = entry.otherModeH;
                desc.colorCombiner.L = entry.combinerL;
                desc.colorCombiner.H = entry.combinerH;
                desc.flags.value = entry.flags;
                shaderHashes[desc.hash()] = true;
                prebuildIndices[desc.hash()] = i;
            }
        }

        prebuildCount = RT64PrebuiltShaderCount;
        prebuildWaitCount = specializedShaders ? RT64PrebuiltShaderWaitCount : RT64PrebuiltShaderCount;
        prebuildState.assign(prebuildCount, PrebuildQueued);
        prebuildBusy = true;
        const uint32_t threads = std::clamp(std::thread::hardware_concurrency() / 2U, 1U, 8U);
        for (uint32_t t = 0; t < threads; t++) {
            prebuildThreads.emplace_back(&RasterShaderCache::prebuildLoop, this, t);
        }

        printf("RT64: %u prebuilt shaders, the game waiting for %u, built %u at a time once the game draws\n", prebuildCount, prebuildWaitCount, threads);
#   endif
    }

    void RasterShaderCache::reportPrebuild(uint32_t done) {
#   if defined(__EMSCRIPTEN__)
        MAIN_THREAD_ASYNC_EM_ASM({ if (Module['rt64ShaderProgress']) Module['rt64ShaderProgress']($0, $1); }, done, prebuildWaitCount);
#   endif
    }

    void RasterShaderCache::prebuildLoop(uint32_t thread) {
#   if defined(RT64_PREBUILT_SHADERS)
        Thread::setCurrentThreadName("RT64 Prebuild");
        while (true) {
            uint32_t index = UINT32_MAX;
            {
                std::unique_lock<std::mutex> lock(prebuildMutex);

                // Once the game is past the wait, one thread builds the rest.
                if ((thread > 0) && (prebuildWaitDone >= prebuildWaitCount)) {
                    break;
                }

                while (!prebuildUrgent.empty() && (index == UINT32_MAX)) {
                    if (prebuildState[prebuildUrgent.front()] != PrebuildTaken) {
                        index = prebuildUrgent.front();
                    }

                    prebuildUrgent.pop();
                }

                while ((index == UINT32_MAX) && (prebuildNext < prebuildCount)) {
                    if (prebuildState[prebuildNext] != PrebuildTaken) {
                        index = prebuildNext;
                    }

                    prebuildNext++;
                }

                if (index == UINT32_MAX) {
                    break;
                }

                prebuildState[index] = PrebuildTaken;
            }

            const RT64PrebuiltShader &entry = RT64PrebuiltShaderList[index];
            ShaderDescription desc;
            memset(&desc, 0, sizeof(desc));
            desc.otherMode.L = entry.otherModeL;
            desc.otherMode.H = entry.otherModeH;
            desc.colorCombiner.L = entry.combinerL;
            desc.colorCombiner.H = entry.combinerH;
            desc.flags.value = entry.flags;
            const char *text = prebuildText.data();
            const bool dualSource = device->getCapabilities().dualSourceBlending;
            const uint32_t pixelOffset = dualSource ? entry.pixelOffset : entry.pixelSingleOffset;
            const uint32_t pixelSize = dualSource ? entry.pixelSize : entry.pixelSingleSize;
            std::unique_ptr<RasterShader> shader = std::make_unique<RasterShader>(device, desc, shaderUber->pipelineLayout.get(), multisampling,
                text + entry.vertexOffset, entry.vertexSize, text + pixelOffset, pixelSize);

            {
                const std::unique_lock<std::mutex> lock(GPUShadersMutex);
                GPUShaders[desc.hash()] = std::move(shader);
            }

            const bool waitedFor = (index < prebuildWaitCount);
            uint32_t waitDone = 0;
            {
                std::unique_lock<std::mutex> lock(prebuildMutex);
                if (waitedFor) {
                    waitDone = ++prebuildWaitDone;
                    if (waitDone == prebuildWaitCount) {
                        printf("RT64: the %u prebuilt shaders the game waits for built in %.1f s\n", prebuildWaitCount, prebuildTimer.elapsedSeconds());
                    }
                }

                if (++prebuildDone == prebuildCount) {
                    prebuildText = std::string();
                    prebuildBusy = false;
                    printf("RT64: %u prebuilt shaders built in %.1f s\n", prebuildCount, prebuildTimer.elapsedSeconds());
                }
            }

            prebuildChanged.notify_all();
            if (waitedFor) {
                reportPrebuild(waitDone);
            }
        }
#   endif
    }

    void RasterShaderCache::waitForPrebuild() {
        // Until something has drawn, no WebGPU layout has been resolved for the prebuilt shaders to be built for.
        if (!prebuildDrawn) {
            return;
        }

        std::unique_lock<std::mutex> lock(prebuildMutex);
        prebuildChanged.wait(lock, [this]() { return prebuildWaitDone >= prebuildWaitCount; });
    }

    // A draw asked for a prebuilt shader that isn't built yet: it is built next.
    void RasterShaderCache::hurryPrebuild(uint64_t shaderHash) {
        std::unique_lock<std::mutex> lock(prebuildMutex);
        auto indexIt = prebuildIndices.find(shaderHash);
        if ((indexIt != prebuildIndices.end()) && (prebuildState[indexIt->second] == PrebuildQueued)) {
            prebuildState[indexIt->second] = PrebuildHurried;
            prebuildUrgent.push(indexIt->second);
        }
    }

    void RasterShaderCache::submit(const ShaderDescription &desc) {
        if (!specializedShaders && !recordShaders) {
            return;
        }

        {
            std::unique_lock<std::mutex> queueLock(submissionMutex);

            // Verify if an entry with the same hash was already submitted before.
            const uint64_t shaderHash = desc.hash();
            bool &found = shaderHashes[shaderHash];
            if (found) {
                return;
            }

            found = true;
        }

        // The spec constants' order: other mode L and H, color combiner L and H, flags.
        if (recordShaders) {
            printf("rt64 shader: %08x %08x %08x %08x %08x\n", desc.otherMode.L, desc.otherMode.H, desc.colorCombiner.L, desc.colorCombiner.H, desc.flags.value);
        }

        if (!specializedShaders) {
            return;
        }

        // Push a new shader compilation to the queue.
        {
            const std::unique_lock<std::mutex> queueLock(descQueueMutex);
            descQueue.push(desc);
        }

        descQueueChanged.notify_all();
    }
    
    void RasterShaderCache::waitForAll() {
        {
            std::unique_lock<std::mutex> queueLock(descQueueMutex);
            descQueue = std::queue<ShaderDescription>();
        }

        bool keepWaiting = false;
        do {
            std::unique_lock<std::mutex> queueLock(descQueueMutex);
            keepWaiting = (descQueueActiveCount > 0);
        } while (keepWaiting);
    }

    void RasterShaderCache::destroyAll() {
        {
            std::unique_lock<std::mutex> lock(GPUShadersMutex);
            GPUShaders.clear();
        }

        {
            std::unique_lock<std::mutex> queueLock(submissionMutex);
            shaderHashes.clear();
        }
    }

    RasterShader *RasterShaderCache::getGPUShader(const ShaderDescription &desc) {
        const uint64_t shaderHash = desc.hash();

        const std::unique_lock<std::mutex> lock(GPUShadersMutex);
        auto shaderIt = GPUShaders.find(shaderHash);
        if (shaderIt == GPUShaders.end()) {
            // The first draw starts the prebuilt shaders' builds (startPrebuild()), and the page shows their progress
            // from there: nothing the launcher draws uses them.
            if ((prebuildCount > 0) && !prebuildDrawn.exchange(true)) {
                prebuildTimer.reset();
                reportPrebuild(0);
            }

            if (prebuildBusy) {
                hurryPrebuild(shaderHash);
            }

            return nullptr;
        }

        return shaderIt->second.get();
    }

    RasterShaderUber *RasterShaderCache::getGPUShaderUber() const {
        return shaderUber.get();
    }

    uint32_t RasterShaderCache::shaderCount() {
        std::unique_lock<std::mutex> lock(GPUShadersMutex);
        return GPUShaders.size();
    }
};