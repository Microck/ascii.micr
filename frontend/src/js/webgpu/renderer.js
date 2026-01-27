import { FontAtlas } from './font-atlas.js';
import { COMPUTE_SHADER, RENDER_SHADER } from './shaders.js';

export class WebGPURenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.adapter = null;
        this.device = null;
        this.context = null;
        
        this.fontAtlas = new FontAtlas();
        this.atlasTexture = null;

        this.inputTexture = null;
        this.charGridBuffer = null;
        this.paramsBuffer = null;
        this.renderTexture = null;
        
        this.solverPipeline = null;
        this.renderPipeline = null;
        
        this.charWidth = 12;
        this.charHeight = 24;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported.");
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw new Error("No WebGPU adapter found.");
        }

        this.device = await this.adapter.requestDevice();

        this.context = this.canvas.getContext('webgpu');

        this.context.configure({
            device: this.device,
            format: 'rgba8unorm',
            alphaMode: 'premultiplied',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        });

        const atlasBitmap = await this.fontAtlas.generate();
        this.createAtlasTexture(atlasBitmap);

        console.log("WebGPU Initialized");
    }

    setSettings(settings) {
        this.charWidth = settings.charWidth;
        this.charHeight = settings.charHeight;
    }

    createAtlasTexture(bitmap) {
        this.atlasTexture = this.device.createTexture({
            size: [bitmap.width, bitmap.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });

        try {
            this.device.queue.copyExternalImageToTexture(
                { source: bitmap },
                { texture: this.atlasTexture },
                [bitmap.width, bitmap.height]
            );
        } catch (error) {
            throw new Error(`Failed to copy font atlas to texture: ${error.message}`);
        }
    }

    async loadImage(url) {
        const img = new Image();
        img.src = url;
        await new Promise(r => img.onload = r);
        
        const bitmap = await createImageBitmap(img);
        this.setupResources(bitmap);
    }

    setupResources(bitmap) {
        const width = bitmap.width;
        const height = bitmap.height;
        
        this.canvas.width = width;
        this.canvas.height = height;

        const gridW = Math.ceil(width / this.charWidth);
        const gridH = Math.ceil(height / this.charHeight);

        try {
            // Input Texture (for compute shader - use rgba32float format)
            this.inputTexture = this.device.createTexture({
                size: [width, height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
            });

            this.device.queue.copyExternalImageToTexture(
                { source: bitmap },
                { texture: this.inputTexture },
                [width, height]
            );

        this.renderTexture = this.device.createTexture({
            size: [width, height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        });

            const gridSize = gridW * gridH * 4;
            this.charGridBuffer = this.device.createBuffer({
                size: gridSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });

            const paramsData = new Float32Array([
                width, height,
                this.charWidth, this.charHeight,
                gridW, gridH
            ]);

            this.paramsBuffer = this.device.createBuffer({
                size: paramsData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.paramsBuffer.getMappedRange()).set(paramsData);
            this.paramsBuffer.unmap();

            this.createPipelines();
        } catch (error) {
            throw new Error(`Failed to setup GPU resources: ${error.message}`);
        }
    }

    createPipelines() {
        const solverModule = this.device.createShaderModule({ code: COMPUTE_SHADER });
        const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });

        this.solverPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: solverModule, entryPoint: 'main' }
        });

        this.solverBindGroup = this.device.createBindGroup({
            layout: this.solverPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.inputTexture.createView() },
                { binding: 1, resource: this.atlasTexture.createView() },
                { binding: 2, resource: { buffer: this.charGridBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        this.renderPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: renderModule, entryPoint: 'main' }
        });

        this.createRenderBindGroup();
    }

    createRenderBindGroup() {
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.renderTexture.createView() },
                { binding: 1, resource: this.atlasTexture.createView() },
                { binding: 2, resource: { buffer: this.charGridBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });
    }

    render() {
        if (!this.solverPipeline || !this.renderPipeline) return;

        const commandEncoder = this.device.createCommandEncoder();

        const solverPass = commandEncoder.beginComputePass();
        solverPass.setPipeline(this.solverPipeline);
        solverPass.setBindGroup(0, this.solverBindGroup);

        const gridW = Math.ceil(this.canvas.width / this.charWidth);
        const gridH = Math.ceil(this.canvas.height / this.charHeight);
        solverPass.dispatchWorkgroups(Math.ceil(gridW / 16), Math.ceil(gridH / 16));
        solverPass.end();

        const renderPass = commandEncoder.beginComputePass();
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.dispatchWorkgroups(Math.ceil(this.canvas.width / 16), Math.ceil(this.canvas.height / 16));
        renderPass.end();

        const canvasTexture = this.context.getCurrentTexture();
        commandEncoder.copyTextureToTexture(
            { texture: this.renderTexture },
            { texture: canvasTexture },
            [this.canvas.width, this.canvas.height]
        );

        this.device.queue.submit([commandEncoder.finish()]);
    }

    async getText() {
        const size = this.charGridBuffer.size;
        const stagingBuffer = this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.charGridBuffer, 0,
            stagingBuffer, 0,
            size
        );
        this.device.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(stagingBuffer.getMappedRange());
        
        let text = "";
        const gridW = Math.ceil(this.canvas.width / this.charWidth);
        const gridH = Math.ceil(this.canvas.height / this.charHeight);

        for (let y = 0; y < gridH; y++) {
            for (let x = 0; x < gridW; x++) {
                const code = data[y * gridW + x];
                text += String.fromCharCode(code || 32); 
            }
            text += '\n';
        }

        stagingBuffer.unmap();
        return text;
    }
}
