# mothertoken — web

Astro static site for [mothertoken.dev](https://mothertoken.dev).

## Stack

- [Astro](https://astro.build) — static site generator, zero JS by default
- IBM Plex Sans + IBM Plex Mono — typography
- No framework, no CSS library — plain scoped CSS per component
- Benchmark data baked in at build time from `src/data/benchmark.ts`
- Deployable on Cloudflare Pages or Vercel (free tier)

## Project structure

```
src/
  data/
    benchmark.ts        ← adapter for src/mothertoken/data/benchmark.json
  layouts/
    Base.astro          ← HTML shell, meta tags
  components/
    Nav.astro           ← tab navigation
    Footer.astro        ← corpus + license info
  pages/
    index.astro         ← home: locale detection + ranked table + calculator
    benchmark.astro     ← full benchmark explorer by tokenizer
    cli.astro           ← CLI usage examples
    methodology.astro   ← corpus, metrics, integrity, reproducibility
  styles/
    global.css          ← design tokens, fonts, shared utilities
```

## Development

```bash
pnpm install
pnpm run dev
```

## Updating benchmark data

Benchmark regeneration is documented in [`../docs/benchmarking.md`](../docs/benchmarking.md). The web app imports `src/mothertoken/data/benchmark.json` at build time.

## Deployment

```bash
pnpm run build   # outputs to dist/
```

Push to GitHub → Cloudflare Pages or Vercel picks it up automatically.

Set the build command to `pnpm run build` and the output directory to `dist`.

## Design decisions

- **No backend** — all data is baked in at build time. The locale detection uses `navigator.language` client-side.
- **No pricing data** — the site shows RTC multipliers only. Users apply them to their own token costs.
- **IBM Plex typeface** — technical, precise, not generic. Mono for data/code, Sans for prose.
- **Dark mode** — supported via `prefers-color-scheme` media query in `global.css`.
