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
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
        });

        // Load Font Atlas
        const atlasBitmap = await this.fontAtlas.generate();
        this.createAtlasTexture(atlasBitmap);

        console.log("WebGPU Initialized");
    }

    createAtlasTexture(bitmap) {
        this.atlasTexture = this.device.createTexture({
            size: [bitmap.width, bitmap.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.atlasTexture },
            [bitmap.width, bitmap.height]
        );
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
        
        // Resize canvas to match image
        this.canvas.width = width;
        this.canvas.height = height;

        // Calculate grid dimensions
        const gridW = Math.ceil(width / this.charWidth);
        const gridH = Math.ceil(height / this.charHeight);

        // Input Texture
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

        const gridSize = gridW * gridH * 4; 
        this.charGridBuffer = this.device.createBuffer({
            size: gridSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
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
        
        const canvasTexture = this.context.getCurrentTexture();
        const renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: canvasTexture.createView() },
                { binding: 1, resource: this.atlasTexture.createView() },
                { binding: 2, resource: { buffer: this.charGridBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.dispatchWorkgroups(Math.ceil(this.canvas.width / 16), Math.ceil(this.canvas.height / 16));
        renderPass.end();

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
