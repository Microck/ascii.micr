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

## features

*   **gradient descent**: uses pytorch and adamw optimizer for high-quality ascii art.
*   **privacy first**: 100% offline processing. no images leave your computer.
*   **y2k aesthetic**: vaporwave colors, retro ui, pixel fonts.
*   **export ready**: copy text to clipboard or save as png.
*   **customizable**: adjustable iterations, learning rate, diversity weights, grid size, and more.
*   **cross-platform**: works on linux, mac, and windows.

## quick start

requires python 3.8+.

```bash
# clone repo
git clone https://github.com/microck/ascii.micr.git

# run the launcher
./run.sh        # linux/mac
# or
run.bat         # windows
```

the browser will open automatically at `http://localhost:8501`

## how it works

1.  **font atlas**: generates a texture atlas from the selected font (cp437/ascii) on the fly.
2.  **render**: composites characters using font bitmaps.
3.  **compare**: calculates mean squared error vs target image.
4.  **optimize**: uses gradient descent (adamw) to adjust character placement.
5.  **iterate**: gradually improves until convergence.

unlike traditional ascii converters that map pixel brightness to characters, this approach **optimizes globally** for the best visual match using gradient descent.

## parameters

*   **iterations**: training steps (100-20000)
*   **learning rate**: gradient descent step size (0.001-0.1)
*   **diversity weight**: penalty for repetitive characters (0.0-0.1)
*   **temperature**: controls sharpness of character selection (0.1 start → 0.01 end)
*   **grid size**: adjustable character width/height and grid dimensions
*   **row gap**: space between rows in pixels (0-20)
*   **optimize alignment**: learn image alignment for better fit
*   **dark mode**: invert colors (white text on black)
*   **encoding**: cp437 (extended) or ascii (standard)

## tips

*   **iterations**: 2000 for quick testing, 10000+ for high quality
*   **diversity**: 0.01 for balanced, 0.0 for accuracy, 0.02+ for artistic
*   **alignment**: enable for better edge matching
*   **image size**: start with images around 500-1000px wide

## development

manual installation:

```bash
# clone repo
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

## tech stack

*   **engine**: pytorch (gradient descent optimization)
*   **framework**: streamlit (web interface)
*   **algorithm**: based on gradscii-art by zellic
*   **style**: y2k aesthetic, css custom properties
*   **fonts**: silkscreen, pixelify sans, tektur, bitarray-a2

## attribution

based on **gradscii-art** by zellic (https://github.com/stong/gradscii-art) - agpl-3.0

see [ATTRIBUTION.md](ATTRIBUTION.md) for complete credits.

## license

agpl-3.0. see [LICENSE](LICENSE) for details.
