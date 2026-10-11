//
// RT64
//

#pragma once

#include "hle/rt64_workload.h"

#include "rt64_descriptor_sets.h"

namespace RT64 {
#if RT64_PACKED_RSP_INPUTS
    // WebGPU guarantees only 8 storage buffers per shader stage, and RSPProcessCS reads 17 inputs and RSPWorldCS 7. So
    // those shaders, built with PACKED_RSP_INPUTS, read all their inputs from one buffer: right before the dispatch,
    // each input's buffer is copied into it whole, one after the other, and the shader gets where each one starts in
    // its push constants. The inputs keep their own buffers, uploads and other readers, so nothing else changes, for
    // a few hundred KB of copies a frame. The RT64_PACKED_RSP_INPUTS build does this on every device, to test it.
    struct RSPPackedInputs {
        struct Copy {
            const RenderBuffer *source;
            uint64_t offset;
            uint64_t size;
        };

        std::unique_ptr<RenderBuffer> buffer;
        uint64_t allocatedSize = 0;
        std::vector<Copy> copies;

        // Lays out the inputs, writes where each one starts into offsets, and makes the buffer big enough to hold them.
        // An input's allocated size is a multiple of 256 (BufferUploader), so every offset is aligned for any load.
        void update(RenderDevice *device, const BufferPair *const *inputs, uint32_t *offsets, uint32_t inputCount) {
            copies.clear();
            uint64_t size = 0;
            for (uint32_t i = 0; i < inputCount; i++) {
                offsets[i] = uint32_t(size);
                if (inputs[i]->get() != nullptr) {
                    copies.push_back({ inputs[i]->get(), size, inputs[i]->allocatedSize });
                    size += inputs[i]->allocatedSize;
                }
            }

            if (allocatedSize < size) {
                allocatedSize = (((size * 3) / 2) + 255) & ~uint64_t(255);
                buffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(allocatedSize, RenderBufferFlag::STORAGE));
            }
        }

        void recordCopies(RenderCommandList *commandList) const {
            commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(buffer.get(), RenderBufferAccess::WRITE));
            for (const Copy &copy : copies) {
                commandList->copyBufferRegion(buffer->at(copy.offset), copy.source->at(0), copy.size);
            }

            commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(buffer.get(), RenderBufferAccess::READ));
        }
    };
#endif

    struct RSPProcessor {
#   if RT64_PACKED_RSP_INPUTS
        static const uint32_t ProcessInputCount = 17;
#   endif

        struct ProcessCB {
            uint32_t vertexStart;
            uint32_t vertexCount;
            float prevFrameWeight;
            float curFrameWeight;
#       if RT64_PACKED_RSP_INPUTS
            uint32_t inputOffsets[ProcessInputCount];
#       endif
        };

        struct ModifyCB {
            uint32_t modifyCount;
        };

        struct ProcessParams {
            RenderWorker *worker = nullptr;
            DrawData *drawData = nullptr;
            DrawBuffers *drawBuffers = nullptr;
            OutputBuffers *outputBuffers = nullptr;
            float prevFrameWeight = 0.0f;
            float curFrameWeight = 1.0f;
        };

        ProcessCB processCB;
        ModifyCB modifyCB;
        std::unique_ptr<RSPProcessDescriptorSet> processSet;
        std::unique_ptr<RSPModifyDescriptorSet> modifySet;
#   if RT64_PACKED_RSP_INPUTS
        RSPPackedInputs packedInputs;
#   endif

        RSPProcessor(RenderDevice *device);
        ~RSPProcessor();
        void process(const ProcessParams &p);
        void recordCommandList(RenderWorker *worker, const ShaderLibrary *shaderLibrary, const OutputBuffers *outputBuffers);
    };
};
