import { defineConfig } from "vite";

// Vercel serves the contents of `dist`. Everything is bundled so the deployed
// page has no runtime network dependency beyond the webfont.
export default defineConfig({
  root: ".",
  build: { outDir: "dist", emptyOutDir: true, target: "es2020", sourcemap: false },
  server: { port: 5173 },
});
