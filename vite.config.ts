import { defineConfig } from "vitest/config";
import wasm from "vite-plugin-wasm";
export default defineConfig({
  plugins: [wasm()],
  test: {
    environment: "node",
    dir: "test",
    watch: false,
    testTimeout: 5000,
    passWithNoTests: true,
    reporters: ["verbose"],
    deps: {
      interopDefault: true,
      registerNodeLoader: true,
    },
  },
});
