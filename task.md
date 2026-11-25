# DSwinIR Studio — Impeccable Upgrade

## Design Context

### Users
- Mobile computer-vision researchers, developers, and photographers evaluating on-device DSwinIR image restoration (de-raining, denoising, deblurring, dehazing, low-light enhancement) and downstream YOLO11 object detection / instance segmentation on Android 10+ devices.
- Context: Testing single photos, inspecting fine pixel-level restoration artifacts, running batch ZIP evaluations (mAP), and benchmarking CPU vs GPU tile latency (5 warm-up + 50 runs).

### Brand Personality
- **3-word personality**: Precise, Optical, Tactile.
- **Voice & Tone**: Clear, technical yet approachable Vietnamese UX copy—concise labels, transparent system state, honest hardware caveats, zero filler.
- **Emotional Goals**: Confidence in measurement fidelity, visual delight when scrubbing Before/After restoration results, calm clarity during long on-device inference runs.

### Aesthetic Direction
- **Visual Direction**: **Swiss Optical Lab** — inspired by precision optical instruments (Leica rangefinders, darkroom loupes, Swiss editorial grids).
- **Theme**: Adaptive **Light & Dark Mode** following system configuration (`values/` + `values-night/`), using tinted archival neutrals (never pure `#000000` or `#FFFFFF` on large surfaces):
  - *Light*: Warm archival paper (`#F6F4EF`), elevated ivory (`#FCFBF8`), deep slate-ink (`#161A1E`), hairline rules (`#DFDDD6`).
  - *Dark*: Darkroom graphite (`#111316`), elevated charcoal (`#1A1D22`), warm silver ink (`#EDEBE6`), subtle border rules (`#2C313A`).
  - *Accents*: Optical Vermilion (`#D64218` light / `#EE6842` dark) for primary action & comparator reticle; Precision Optical Teal (`#0E7C7B` light / `#3BB2B0` dark) for telemetry, GPU status, and restored-state indicators.
- **Anti-references**: Generic AI neon-on-dark cyber-slop, purple-blue gradients, glassmorphism cards, nested card-inside-card boxes, sterile boilerplate Material blue dialogs.

### Design Principles
1. **Image-First Hierarchy**: The photo and Before/After split comparator command visual attention; chrome stays tight and purposeful.
2. **Flat Precision over Card Soup**: Use 1dp hairline dividers, intentional whitespace, and subtle surface shifts instead of heavy rounded white cards on gray backgrounds.
3. **Transparent Telemetry**: Show tile progress, elapsed time, CPU/GPU latency comparisons, and mAP deltas with tabular numbers (`tnum`) and visual bars rather than walls of plain text.
4. **Purposeful Motion & Tactility**: Smooth ease-out-quint transitions (`cubic-bezier(0.22, 1, 0.36, 1)`), subtle press scale feedback, haptic detents on the Before/After slider, respecting system animator scale.
5. **Test & Contract Integrity**: Preserve all Android view contracts (`Button` labels, `onActivityResult` flow, `ActionReadinessDeviceTest`, `RestorationDeviceTest`, `PipelineTest`).
