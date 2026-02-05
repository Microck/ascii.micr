<div align="center">
  <img src="https://github.com/user-attachments/assets/0b497c2b-6cbf-4da6-88c0-8dd02179f007" width="300" alt="ascii.micr logo" />
  
  <h1>ascii.micr</h1>
  
  <p><strong>local python ascii art generator. gradient descent optimization. 100% offline.</strong></p>

  <p>
    <a href="https://opensource.org/licenses/AGPL-3.0">
      <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg?style=flat-square" alt="license" />
    </a>
    <a href="https://pytorch.org">
      <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg?style=flat-square" alt="pytorch" />
    </a>
    <a href="https://streamlit.io">
      <img src="https://img.shields.io/badge/Streamlit-1.0+-red.svg?style=flat-square" alt="streamlit" />
    </a>
  </p>

  <br />

  <img src="https://github.com/user-attachments/assets/dd8d034b-8a57-40a7-b56b-bd571e2e18e6" width="800" alt="application interface preview" />
</div>

<br />

## ⚠️ Important Update

**The WebGPU version is deprecated.** This project has been rebuilt as a **local Python application** using Streamlit. The web version had browser compatibility issues and required specific WebGPU support that wasn't widely available.

The new local version:
- ✅ **Works on all platforms** (Linux, Mac, Windows)
- ✅ **No browser required** - runs in your browser locally via Streamlit
- ✅ **More reliable** - uses proven PyTorch instead of experimental WebGPU
- ✅ **Same algorithm** - still uses gradient descent optimization
- ✅ **Better performance** - uses your CPU/GPU efficiently
- ✅ **100% offline** - no internet connection needed after installation

<br />

## features

*   **gradient descent optimization**: uses pytorch and adamw optimizer for high-quality ascii art.
*   **local processing**: 100% offline. no images uploaded to any server.
*   **y2k aesthetic**: vaporwave colors, retro ui, pixel fonts.
*   **export ready**: copy text to clipboard or save as png.
*   **customizable**: adjustable iterations, learning rate, diversity weights, grid size, and more.
*   **cross-platform**: works on linux, mac, and windows.

<br />

## quick start

### option 1: easy install (recommended)

```bash
# clone the repository
git clone https://github.com/microck/ascii.micr.git
cd ascii.micr

# run the launcher
./run.sh        # linux/mac
# or
run.bat         # windows
```

that's it! the browser will open automatically at `http://localhost:8501`

### option 2: manual install

```bash
# clone repository
git clone https://github.com/microck/ascii.micr.git
cd ascii.micr

# create virtual environment
python3 -m venv venv

# activate virtual environment
source venv/bin/activate  # linux/mac
# or
venv\Scripts\activate     # windows

# install dependencies
pip install -r requirements.txt

# run the application
streamlit run app.py
```

<br />

## how it works

this tool uses **gradient descent optimization** instead of traditional lookup tables:

1.  **random start**: begins with random character selections across a grid.
2.  **render**: composites characters using font bitmaps.
3.  **compare**: calculates mean squared error vs target image.
4.  **optimize**: uses gradient descent (adamw) to adjust character placement.
5.  **iterate**: gradually improves until convergence.

unlike traditional ascii converters that map pixel brightness to characters, this approach **optimizes globally** for the best visual match.

<br />

## parameters

### grid configuration
- **character width/height**: size of each character (8-16px / 16-32px)
- **grid width/height**: number of characters per row/column (20-80 / 10-40)
- **row gap**: space between rows in pixels (0-20px)

### training parameters
- **iterations**: training steps (100-20000)
- **learning rate**: gradient descent step size (0.001-0.1)
- **diversity weight**: penalty for repetitive characters (0.0-0.1)

### temperature settings
- **temp start**: initial temperature for soft character selection (0.1-5.0)
- **temp end**: final temperature for discrete character selection (0.001-1.0)

### options
- **optimize alignment**: learn image alignment for better fit
- **use gumbel softmax**: add noise for exploration during training
- **dark mode**: invert colors (white text on black background)
- **encoding**: cp437 (extended ascii) or ascii (standard)

<br />

## tips for best results

1. **image size**: start with images around 500-1000px wide
2. **iterations**: 2000 for quick testing, 10000+ for high quality
3. **diversity**: 0.01 for balanced output, 0.0 for accuracy, 0.02+ for artistic effect
4. **alignment**: enable "optimize alignment" for better edge matching
5. **dark mode**: toggle based on your target background

<br />

## tech stack

*   **engine**: pytorch (gradient descent optimization)
*   **framework**: streamlit (web interface)
*   **algorithm**: based on gradscii-art by zellic
*   **style**: y2k aesthetic, css custom properties
*   **fonts**: silkscreen, pixelify sans, tektur, bitarray-a2

<br />

## project structure

```
ascii.micr/
├── app.py                 # main streamlit application
├── run.sh                 # linux/mac launcher
├── run.bat                # windows launcher
├── requirements.txt       # python dependencies
├── LICENSE               # agpl-3.0 license
├── ATTRIBUTION.md        # credits and attribution
├── README.md             # this file
├── .streamlit/           # streamlit configuration
└── gradscii-art/         # ascii art engine (submodule)
    ├── train.py
    ├── fonts/
    └── testcases/
```

<br />

## attribution

this project is based on:

*   **gradscii-art** by zellic - [github.com/stong/gradscii-art](https://github.com/stong/gradscii-art) (agpl-3.0)
*   original **ascii.micr** webgpu concept by microck (mit)

see [ATTRIBUTION.md](ATTRIBUTION.md) for complete credits.

<br />

## license

this project is licensed under the **gnu affero general public license v3.0 (agpl-3.0)**.

this program is free software: you can redistribute it and/or modify it under the terms of the gnu affero general public license as published by the free software foundation, either version 3 of the license, or (at your option) any later version.

see [LICENSE](LICENSE) for full details.

<br />

## troubleshooting

### "module not found" errors
make sure the virtual environment is activated:
```bash
source venv/bin/activate  # linux/mac
venv\Scripts\activate     # windows
```

### slow performance
- reduce iterations for faster results
- use cpu-only pytorch if you don't have a gpu:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### fonts not loading
ensure the `gradscii-art/fonts/` directory exists and contains the font files.

<br />

---

**made with 💖 and gradient descent**

*[previous webgpu version available in git history]*
