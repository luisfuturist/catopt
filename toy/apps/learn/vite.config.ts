import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite-plus";

export default defineConfig({
  plugins: [tailwindcss()],
  resolve: {
    alias: {
      engine: fileURLToPath(new URL("../../packages/engine/src/index.ts", import.meta.url)),
    },
  },
});
