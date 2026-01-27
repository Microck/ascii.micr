import { WebGPURenderer } from './webgpu/renderer.js';

const ASCII_RAMP = " .:-=+*#%@";
const CP437_RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

function getAsciiRamp(encoding) {
    if (encoding === 'ascii') {
        return ASCII_RAMP;
    }
    return CP437_RAMP;
}

function computeAsciiFromImage(img, options) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    const maxDim = options.maxDim;
    const scale = Math.min(1, maxDim / width, maxDim / height);
    const targetWidth = Math.max(1, Math.floor(width * scale));
    const targetHeight = Math.max(1, Math.floor(height * scale));
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

    const data = ctx.getImageData(0, 0, targetWidth, targetHeight).data;
    const baseRamp = getAsciiRamp(options.encoding);
    const rampSize = Math.max(2, Math.round(baseRamp.length * options.rampScale));
    const ramp = baseRamp.slice(0, rampSize);
    const cw = options.charWidth;
    const ch = options.charHeight;
    const extraLines = Math.max(0, Math.round(options.rowGap / 6));
    let text = "";

    for (let y = 0; y < targetHeight; y += ch) {
        for (let x = 0; x < targetWidth; x += cw) {
            let sum = 0;
            let count = 0;
            for (let py = 0; py < ch && y + py < targetHeight; py++) {
                for (let px = 0; px < cw && x + px < targetWidth; px++) {
                    const i = ((y + py) * targetWidth + (x + px)) * 4;
                    const r = data[i] / 255;
                    const g = data[i + 1] / 255;
                    const b = data[i + 2] / 255;
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    lum = Math.min(1, Math.max(0, lum + options.bias));
                    lum = Math.pow(lum, 1 / options.contrast);
                    sum += lum;
                    count++;
                }
            }
            const avg = count ? sum / count : 0;
            const idx = Math.max(0, Math.min(ramp.length - 1, Math.floor((1 - avg) * (ramp.length - 1))));
            text += ramp[idx];
        }
        text += "\n";
        for (let i = 0; i < extraLines; i++) {
            text += "\n";
        }
    }

    return text;
}

function withTimeout(promise, ms) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            reject(new Error("timeout"));
        }, ms);
        promise.then(result => {
            clearTimeout(timer);
            resolve(result);
        }).catch(error => {
            clearTimeout(timer);
            reject(error);
        });
    });
}

async function ensureImageReady(img) {
    if (img.complete && img.naturalWidth) {
        return;
    }
    if (img.decode) {
        try {
            await img.decode();
            return;
        } catch {
        }
    }
    await new Promise(resolve => {
        img.onload = () => resolve();
        img.onerror = () => resolve();
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    
    const browseBtn = document.getElementById('browse-btn');
    const realInput = document.getElementById('real-file-input');
    const filePathDisplay = document.getElementById('file-path');
    const goBtn = document.getElementById('go-btn');
    const demoBtn = document.getElementById('demo-btn');
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
    const forceCpu = new URLSearchParams(window.location.search).has('cpu');
    let rendererReady = false;
    try {
        if (!forceCpu) {
            await renderer.init();
            console.log("Renderer ready");
            rendererReady = true;
        }
    } catch (e) {
        outputWindow.classList.remove('hidden');
        asciiOutput.textContent = "WebGPU failed to initialize. Using CPU fallback.";
    }

    window.addEventListener('error', (event) => {
        outputWindow.classList.remove('hidden');
        asciiOutput.textContent = `ERROR: ${event.message}`;
    });

    window.addEventListener('unhandledrejection', (event) => {
        outputWindow.classList.remove('hidden');
        asciiOutput.textContent = `ERROR: ${event.reason && event.reason.message ? event.reason.message : 'Unhandled rejection'}`;
    });

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

    function getSettings() {
        const iterations = Number(params.iterations.input.value);
        const lr = Number(params.lr.input.value);
        const diversity = Number(params.diversity.input.value);
        const tempStart = Number(params.tempStart.input.value);
        const tempEnd = Number(params.tempEnd.input.value);
        const rowGap = Number(params.rowGap.input.value);
        const encoding = params.encoding.input.value;
        const optimize = params.optimizeAlignment.input.checked;

        const maxDim = Math.round(128 + (iterations - 100) / (20000 - 100) * 896);
        const contrast = Math.min(2, Math.max(0.6, 0.6 + lr * 8));
        const rampScale = Math.min(1, Math.max(0.35, 0.4 + diversity * 6));
        const bias = (tempStart - tempEnd) * 0.02;
        const charWidth = optimize ? 10 : 12;
        const charHeight = optimize ? 20 : 24;

        return {
            maxDim,
            contrast,
            rampScale,
            bias,
            charWidth,
            charHeight,
            rowGap,
            encoding
        };
    }

    function adaptOutputWindow(text) {
        const lines = text.split('\n');
        const rows = Math.max(1, lines.length);
        const cols = Math.max(1, ...lines.map(line => line.length));
        const charWidth = 8;
        const lineHeight = 14;
        const paddingX = 60;
        const paddingY = 140;
        const maxWidth = Math.min(window.innerWidth * 0.6, 900);
        const maxHeight = Math.min(window.innerHeight * 0.6, 520);
        const targetWidth = Math.min(maxWidth, Math.max(260, cols * charWidth + paddingX));
        const targetHeight = Math.min(maxHeight, Math.max(180, rows * lineHeight + paddingY));

        outputWindow.style.flex = '0 0 auto';
        outputWindow.style.width = `${Math.round(targetWidth)}px`;
        outputWindow.style.height = `${Math.round(targetHeight)}px`;
        asciiOutput.style.width = `${Math.round(targetWidth - 40)}px`;
        asciiOutput.style.height = `${Math.round(targetHeight - 120)}px`;
        asciiOutput.style.overflow = 'auto';
    }

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

    async function setDemoFile() {
        const response = await fetch('/favicon.png');
        const blob = await response.blob();
        const file = new File([blob], 'favicon.png', { type: blob.type || 'image/png' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        realInput.files = dataTransfer.files;
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

    demoBtn.addEventListener('click', async () => {
        await setDemoFile();
        goBtn.click();
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
            outputWindow.classList.remove('hidden');
            asciiOutput.textContent = "ERROR: NO SOURCE FILE DETECTED";
            return;
        }

        outputWindow.classList.remove('hidden');
        asciiOutput.textContent = "INITIALIZING WEBGPU CORE...\n";
        goBtn.disabled = true;
        goBtn.textContent = "PROCESSING...";

        try {
            await ensureImageReady(inputPreviewImg);

            const settings = getSettings();
            const fallbackText = computeAsciiFromImage(inputPreviewImg, settings);
            asciiOutput.textContent = fallbackText;
            adaptOutputWindow(fallbackText);
            gpuCanvas.classList.add('hidden');
            asciiOutput.classList.remove('hidden');
            toggleViewBtn.textContent = "VIEW: IMAGE";

            if (!rendererReady) {
                return;
            }

            renderer.setSettings(settings);
            await renderer.loadImage(inputPreviewImg.src);

            asciiOutput.textContent += "\nCOMPUTING TENSOR GRID...\n";

            renderer.render();

            asciiOutput.textContent += "EXTRACTING TEXT BUFFER...\n";

            const text = await withTimeout(renderer.getText(), 4000);
            if (text && text.trim().length > 0) {
                asciiOutput.textContent = text;
                adaptOutputWindow(text);
                gpuCanvas.classList.remove('hidden');
                asciiOutput.classList.add('hidden');
                toggleViewBtn.textContent = "VIEW: TEXT";
            }

        } catch (error) {
            const settings = getSettings();
            const fallbackText = computeAsciiFromImage(inputPreviewImg, settings);
            asciiOutput.textContent = fallbackText;
            adaptOutputWindow(fallbackText);
            gpuCanvas.classList.add('hidden');
            asciiOutput.classList.remove('hidden');
            toggleViewBtn.textContent = "VIEW: IMAGE";
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
