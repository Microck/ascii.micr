import { WebGPURenderer } from './webgpu/renderer.js';

document.addEventListener('DOMContentLoaded', async () => {
    
    const browseBtn = document.getElementById('browse-btn');
    const realInput = document.getElementById('real-file-input');
    const filePathDisplay = document.getElementById('file-path');
    const goBtn = document.getElementById('go-btn');
    const outputWindow = document.getElementById('output-window');
    const asciiOutput = document.getElementById('ascii-output');
    const resultPreviewImg = document.getElementById('result-preview-img');
    const gpuCanvas = document.getElementById('gpu-canvas');
    const toggleViewBtn = document.getElementById('toggle-view-btn');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    
    const inputPreviewArea = document.getElementById('input-preview-area');
    const inputPreviewImg = document.getElementById('input-preview-img');
    const sourcePreviewWindow = document.getElementById('source-preview-window');
    const visualColumn = document.getElementById('visual-column');

    const aboutBtn = document.getElementById('about-btn');
    const panelControls = document.getElementById('panel-controls');
    const panelAbout = document.getElementById('panel-about');
    const backToDashBtn = document.getElementById('back-to-dash-btn');

    const renderer = new WebGPURenderer(gpuCanvas);
    try {
        await renderer.init();
        console.log("Renderer ready");
    } catch (e) {
        alert("WebGPU failed to initialize: " + e.message);
    }

    const params = {
        iterations: { input: document.getElementById('param-iterations'), display: document.getElementById('val-iterations') },
        lr: { input: document.getElementById('param-lr'), display: document.getElementById('val-lr') },
        diversity: { input: document.getElementById('param-diversity'), display: document.getElementById('val-diversity') },
        tempStart: { input: document.getElementById('param-temp-start'), display: document.getElementById('val-temp-start') },
        tempEnd: { input: document.getElementById('param-temp-end'), display: document.getElementById('val-temp-end') },
        optimizeAlignment: { input: document.getElementById('param-optimize-alignment') },
        darkMode: { input: document.getElementById('param-dark-mode') },
        encoding: { input: document.getElementById('param-encoding') },
        rowGap: { input: document.getElementById('param-row-gap'), display: document.getElementById('val-row-gap') }
    };

    aboutBtn.addEventListener('click', () => {
        panelControls.classList.add('hidden');
        panelAbout.classList.remove('hidden');
    });

    backToDashBtn.addEventListener('click', () => {
        panelAbout.classList.add('hidden');
        panelControls.classList.remove('hidden');
    });

    toggleViewBtn.addEventListener('click', () => {
        const isCanvasHidden = gpuCanvas.classList.contains('hidden');
        
        if (isCanvasHidden) {
            asciiOutput.classList.add('hidden');
            gpuCanvas.classList.remove('hidden');
            toggleViewBtn.textContent = "VIEW: TEXT";
        } else {
            gpuCanvas.classList.add('hidden');
            asciiOutput.classList.remove('hidden');
            toggleViewBtn.textContent = "VIEW: IMAGE";
        }
    });

    Object.values(params).forEach(param => {
        if (param.display) {
            param.input.addEventListener('input', () => {
                param.display.textContent = param.input.value;
            });
        }
    });

    browseBtn.addEventListener('click', () => {
        realInput.click();
    });

    realInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            filePathDisplay.value = `C:\\UPLOADS\\${file.name.toUpperCase()}`;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                inputPreviewImg.src = e.target.result;
                inputPreviewImg.classList.remove('hidden');
                document.getElementById('preview-placeholder').classList.add('hidden');
                sourcePreviewWindow.classList.remove('hidden');
                visualColumn.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    goBtn.addEventListener('click', async () => {
        if (!realInput.files.length) {
            alert("ERROR: NO SOURCE FILE DETECTED");
            return;
        }

        outputWindow.classList.remove('hidden');
        asciiOutput.textContent = "INITIALIZING WEBGPU CORE...\n";
        goBtn.disabled = true;
        goBtn.textContent = "PROCESSING...";

        try {
            await renderer.loadImage(inputPreviewImg.src);
            
            asciiOutput.textContent += "COMPUTING TENSOR GRID...\n";
            
            renderer.render();
            
            asciiOutput.textContent += "EXTRACTING TEXT BUFFER...\n";
            
            const text = await renderer.getText();
            asciiOutput.textContent = text;
            
            gpuCanvas.classList.remove('hidden');
            asciiOutput.classList.add('hidden');
            toggleViewBtn.textContent = "VIEW: TEXT";

        } catch (error) {
            asciiOutput.textContent += `\nCRITICAL FAILURE:\n${error.message}`;
            console.error(error);
        } finally {
            goBtn.disabled = false;
            goBtn.textContent = "EXECUTE";
        }
    });

    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(asciiOutput.textContent)
            .then(() => {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = "COPIED!";
                setTimeout(() => copyBtn.textContent = originalText, 2000);
            });
    });

    downloadBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = `ASCII_GPU_${Date.now()}.png`;
        link.href = gpuCanvas.toDataURL();
        link.click();
    });
});
