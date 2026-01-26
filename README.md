<div align="center">
  <img src="frontend/favicon.png" width="100" alt="ascii.micr logo" />
  
  <h1>ascii.micr</h1>
  
  <p><strong>client-side webgpu ascii art generator. gradient descent optimization. zero server uploads.</strong></p>

  <p>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/license-MIT-000000.svg?style=flat-square" alt="license" />
    </a>
    <a href="https://developer.chrome.com/docs/web-platform/webgpu">
      <img src="https://img.shields.io/badge/webgpu-enabled-000000.svg?style=flat-square" alt="webgpu" />
    </a>
    <a href="https://vitejs.dev/">
      <img src="https://img.shields.io/badge/vite-5.0-000000.svg?style=flat-square" alt="vite" />
    </a>
  </p>

  <br />

  <img src="https://ascii.micr.zone/og-preview.png" width="800" alt="application interface preview" />
</div>

<br />

## features

*   **instant render**: uses webgpu compute shaders for l2 distance matching.
*   **privacy first**: 100% client-side processing. no images leave your browser.
*   **y2k aesthetic**: custom kaomoji patterns, scrolling backgrounds, neon palette.
*   **export ready**: copy text to clipboard or save as png.
*   **customizable**: adjustable iterations, learning rate, and diversity weights.

## how it works

1.  **font atlas**: generates a texture atlas from the selected font (cp437/ascii) on the fly.
2.  **compute shader**: runs a parallel l2 distance comparison between input pixels and font glyphs.
3.  **optimization**: finds the best matching character for every grid cell instantly.
4.  **rendering**: outputs directly to a canvas for preview and a text buffer for copying.

## development

requires node.js 18+ and a webgpu-compatible browser (chrome/edge/arc).

```bash
# clone repo
git clone https://github.com/microck/ascii.micr.git

# install dependencies
cd frontend
npm install

# start dev server
npm run dev
```

## tech stack

*   **engine**: webgpu (wgsl shaders)
*   **framework**: vite (vanilla js module)
*   **style**: css variables, svg data uris
*   **fonts**: silkscreen, pixelify sans, tektur

## license

mit
