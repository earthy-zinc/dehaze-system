import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  outDir: "dist",
  external: ["axios"],
  splitting: false,
  esbuildOptions(options) {
    options.alias = {
      "@": "./src",
      "#": "./test",
    };
  },
  target: "es2020",
  platform: "neutral",
});
