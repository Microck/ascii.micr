# gradscii-art Frontend Implementation

## Context

Original Request: Turn gradscii-art into usable frontend
Constraint: Frontend-only, simple like commitmono.com
Decision: User will host backend on own VPS

## Architecture

```
Frontend (Next.js) → HTTP POST → User's VPS (FastAPI + PyTorch) → Response (PNG + text)
```

**Rationale**: Full gradscii functionality requires PyTorch GPU. Browser-only rewrite = 2-3 weeks, inferior quality. VPS = zero marginal cost (user already pays hosting).

## Work Objectives

### Core Objective
Create commitmono.com-style frontend for gradscii-art gradient descent generator with optional backend integration.

### Deliverables
- Next.js frontend (Vercel deployment)
- FastAPI wrapper for train.py
- VPS deployment guide

### Definition of Done
- Frontend generates ASCII using user's VPS OR user's own VPS

---

## Verification Strategy

**Infrastructure**: None (frontend-only deployment)
**Framework**: No tests
**QA**: Manual browser testing

---

## Task Flow

```
Frontend Build → Deploy to Vercel → VPS Deploy → Test Integration
```

## TODOs

### Phase 1: Frontend Setup

- [ ] 1. Initialize Next.js project
  - [ ] Use `npx create-next-app@latest` with TypeScript, Tailwind, App Router
  - [ ] Install: `@radix-ui/react-dialog`, `lucide-react`
  - [ ] Verify: `npm run dev` → localhost:3000 starts

- [ ] 2. Create core components
  - [ ] `components/ImageUploader.tsx` - Drag-drop + file picker + paste
  - [ ] `components/PresetSelector.tsx` - Epson/Discord toggle
  - [ ] `components/ParameterPanel.tsx` - Iterations, LR, diversity sliders
  - [ ] `components/AsciiDisplay.tsx` - Original image + ASCII preview (side-by-side)
  - [ ] `components/ProgressBar.tsx` - Training progress bar
  - [ ] `components/ExportActions.tsx` - Download PNG + Copy text
  - [ ] Parallelizable: YES (1-2, 3-4, 5-6)

- [ ] 3. Implement API integration
  - [ ] `lib/api.ts` - POST to `/generate`, handle streaming progress
  - [ ] `hooks/useGeneration.ts` - State management for generation status
  - [ ] Progress streaming via Server-Sent Events (SSE)
  - [ ] Parallelizable: NO (depends on 0, 1, 2)
  - [ ] **References**:
    - Pattern: Next.js Fetch API docs (`/docs/app/building-your-application/fetching`)
    - API: FastAPI StreamingResponse (`fastapi.tiangolo.com/en/latest/tutorial/response`)
    - External: `github.com/sysid/sse` for SSE patterns

- [ ] 4. Build main page layout
  - [ ] Simple grid: Upload (left) | Preview (right)
  - [ ] **Dark mode by default** (CSS: `bg-zinc-950 text-zinc-50`)
  - [ ] Commit Mono font via CDN
  - [ ] Mobile responsive: Stack controls on tablet, single column on mobile
  - [ ] **Super minimal design** - single page, no complex layouts, clean typography
  - [ ] **Same style as commitmono.com** - Gray (#9ca3af) + Black (#0a0a0a) color scheme
  - [ ] Parallelizable: NO (depends on 3)
  - [ ] **References**:
    - UI: commitmono.com analysis (examined design patterns)
    - Layout: Radix UI Grid docs (`radix-ui.com/primitives/docs/grid`)
    - Font: commitmono.com download instructions

### Phase 2: Backend Wrapper

- [ ] 5. Create FastAPI project
  - [ ] `api/main.py` - Wraps `train.py`
  - [ ] Modify train.py to accept params dict (not argparse)
  - [ ] Save uploaded image to `/tmp`, pass path to train()
  - [ ] Expose progress via callback/SSE
  - [ ] Return outputs as base64 PNG + strings
  - [ ] Parallelizable: NO (depends on 5)
  - [ ] **References**:
    - FastAPI: `fastapi.tiangolo.com/en/latest/tutorial/first-steps/`
    - Pattern: Current gradscii codebase (analyzed train.py)
    - Upload: FastAPI UploadFile docs (`fastapi.tiangolo.com/en/latest/tutorial/request-files/`)

- [ ] 6. Create deployment artifacts
  - [ ] `Dockerfile` - Python 3.10 + PyTorch install
  - [ ] `requirements-api.txt` - FastAPI, uvicorn
  - [ ] `.gitignore` - Exclude `.pyc`, `__pycache__`, `/tmp`
  - [ ] Parallelizable: YES (with 5)
  - [ ] **References**:
    - Docker: `docs.docker.com/get-started/`
    - PyTorch GPU: `pytorch.org/docs/stable/notes/cuda.html`

- [ ] 7. Deploy to user's VPS
  - [ ] SSH into VPS, clone repo, install Docker
  - [ ] Run: `docker run -p 8000:8000 --gpus all gradscii-api`
  - [ ] Configure Nginx reverse proxy (optional for HTTPS)
  - [ ] Parallelizable: NO (sequential)
  - [ ] **Acceptance**:
    - `curl http://your-vps:8000/docs` → FastAPI docs load
    - Test POST with sample image → Returns PNG + text

### Phase 3: Integration & Testing

- [ ] 8. Connect frontend to backend
  - [ ] Environment variable: `NEXT_PUBLIC_API_URL=http://your-vps:8000`
  - [ ] Fallback mode: Frontend-only (show demo with pre-generated outputs)
  - [ ] Parallelizable: NO (depends on 5, 7)
  - [ ] **References**:
    - Next.js Env: `nextjs.org/docs/app/building-your-application/configuring`
    - Error handling: Radix UI Toast docs

- [ ] 9. Manual QA
  - [ ] Test image upload (drag-drop, file picker, clipboard paste)
  - [ ] Test all presets (Epson, Discord)
  - [ ] Test parameter changes (iterations, diversity, temperature)
  - [ ] Verify progress streaming (updates every 100 iters)
  - [ ] Test exports: Download PNG works, Copy to clipboard works
  - [ ] Test frontend-only mode (when backend unreachable)
  - [ ] Parallelizable: YES (with 8)
  - [ ] **Acceptance**:
    - Generate 10,000 iteration job → Verify PNG + text output
    - Check all console.log for errors
    - Test mobile layout

### Phase 4: Deploy Frontend

- [ ] 10. Deploy to Vercel
  - [ ] Connect GitHub repo → Vercel → Deploy
  - [ ] Verify: Domain loads, API env var set
  - [ ] Parallelizable: NO (sequential after testing)
  - [ ] **References**:
    - Vercel: `vercel.com/docs/deployments/overview`
    - Env vars: `vercel.com/docs/projects/environment-variables`

- [ ] 11. Final end-to-end test
  - [ ] Visit deployed URL, test full workflow
  - [ ] Check mobile responsiveness
  - [ ] Verify dark mode
  - [ ] Parallelizable: NO (last task)
  - [ ] **Acceptance**:
    - Upload image → Generate → Download PNG
    - Full cycle works in < 60 seconds (with GPU)

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|--------|
| 4 | `feat(api): fastapi wrapper` | `api/main.py`, `Dockerfile` |
| 7 | `feat(deploy): vps setup` | `.gitignore`, `requirements-api.txt` |
| 10 | `feat(deploy): vercel frontend` | `.env.local`, README deployment |

---

## Success Criteria

### Verification Commands
```bash
# Backend health
curl http://your-vps:8000/docs

# Frontend build
npm run build && npm run start

# Full E2E test (manual)
- Upload image → Click Generate → Download PNG
```

### Final Checklist
- [ ] Frontend generates ASCII art via API
- [ ] Optional: User can deploy own backend
- [ ] Fallback mode works when API unavailable
- [ ] All exports functional
- [ ] Simple UI (commitmono.com style)
- [ ] Dark mode default
- [ ] Mobile responsive

---

## Configuration

### API URL (User Decision Needed)

**Decision**: What is your VPS IP/domain?

- [ ] Set in: `.env.local` → `NEXT_PUBLIC_API_URL=http://YOUR-VPS:8000`
- [ ] Default: Frontend-only demo mode

### Presets

| Name | Encoding | Font | Row Gap |
|------|----------|------|----------|
| Epson | CP437 | bitArray-A2.ttf | 6px |
| Discord | ASCII | gg mono.ttf | 0px |

### Parameters

| Param | Default | Range |
|--------|---------|-------|
| Iterations | 10000 | 1000-20000 |
| Learning Rate | 0.01 | 0.001-0.1 |
| Diversity Weight | 0.01 | 0.0-0.1 |
| Temp Start | 1.0 | 0.1-5.0 |
| Temp End | 0.01 | 0.001-1.0 |

---

## Notes

**Design Principle**: Simplicity over features. Commitmono.com proves minimal UI beats complexity.

**Tech Stack**:
- Frontend: Next.js 15 + TypeScript + Tailwind + Radix UI
- Backend: Python 3.10 + FastAPI + PyTorch 2.10
- Hosting: Vercel (frontend) + User's VPS (backend)

**Performance Expectations**:
- Backend: 10K iterations in 3-5 minutes (M2 with GPU)
- Frontend: Instant preview, < 1s to start generation
- API latency: Dependent on user's VPS location

**Security**:
- API: No auth required for MVP
- Frontend: Rate limiting (100 req/min) to prevent abuse
- Input validation: Max file size 10MB, formats PNG/JPG/WebP

**Questions for User**:
1. What is your VPS domain/IP for API URL?
2. Do you need HTTPS (SSL) setup instructions?
