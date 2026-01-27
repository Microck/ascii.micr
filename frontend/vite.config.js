import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [],
  build: {
    target: 'es2020',
    minify: 'esbuild',
    rollupOptions: {
      output: {
        // Preserve formatting for shader files
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.js')) {
            return assetInfo.name.replace('.js', '.min.js');
          }
          return assetInfo.name;
        },
      }
    }
  }
});
