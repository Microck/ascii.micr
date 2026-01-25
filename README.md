# gradscii-art

Gradient descent ASCII art generator built with Next.js and PyTorch.

## Live Demo

Deployed frontend on Vercel: [View Site](https://your-app.vercel.app)

## Features

- **Image-to-ASCII conversion**: Upload any image and convert to ASCII art
- **Gradient descent optimization**: Uses PyTorch for high-quality output
- **Two presets**: 
  - **Epson**: CP437 encoding with 6px row gap (retail displays)
  - **Discord**: ASCII encoding with 0px row gap (Discord-style)
- **Real-time preview**: Side-by-side comparison of original and ASCII
- **Export options**: Download as PNG or copy text to clipboard
- **Responsive design**: Works on desktop, tablet, and mobile
- **Dark mode default**: Styled after commitmono.com

## Tech Stack

**Frontend**
- Next.js 16.1.4
- React 19.2.3
- TypeScript 5
- Tailwind CSS 4
- Lucide React (icons)

**Backend** (optional)
- FastAPI
- Python 3.10
- PyTorch 2.10.0

## Quick Start

### Local Development

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

### With Backend (Optional)

Backend enables real-time generation. Without backend, app uses demo mode.

```bash
cd api
docker build -t gradscii-api .
docker run -p 8000:8000 --gpus all gradscii-api
```

Set environment variable:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Deployment

### Frontend (Vercel)

1. Connect GitHub repo to Vercel
2. Set environment variable: `NEXT_PUBLIC_API_URL` (optional, for backend connection)
3. Deploy (automatic)

### Backend (Docker on VPS)

```bash
docker build -t gradscii-api .
docker run -d -p 8000:8000 --gpus all gradscii-api
```

## Parameters

- **Iterations**: 1000-20000 (higher = better quality, slower)
- **Learning Rate**: 0.001-0.1 (convergence speed)
- **Diversity Weight**: 0.0-0.1 (character variety)
- **Temp Start**: 0.1-5.0 (initial randomness)
- **Temp End**: 0.001-1.0 (final precision)

## License

MIT

## Acknowledgments

- [commit-mono](https://github.com/eigilnikolajsen/commit-mono) - Programming typeface
- [gradscii-art](https://github.com/microcon/gradscii-art) - ASCII art generation
