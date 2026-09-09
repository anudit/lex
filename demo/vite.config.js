import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { writeFileSync } from 'node:fs';

// `lex` is linked from elsewhere in the repo. The dev server resolves it through
// a node_modules symlink, but Rollup resolves symlinks to their real path, which
// lands outside the Vite root and fails the production build -- so it is aliased
// explicitly. gpu-lexer comes from npm and needs no alias.
const at = (p) => fileURLToPath(new URL(p, import.meta.url));

// Dev-only endpoint so /capture.html can write results.json itself, rather than
// asking a human to copy 13 KB of JSON out of a textarea.
function saveResults() {
  return {
    name: 'save-results',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__save-results', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; return res.end(); }
        const chunks = [];
        req.on('data', (c) => chunks.push(c));
        req.on('end', () => {
          const target = fileURLToPath(new URL('./src/results.json', import.meta.url));
          writeFileSync(target, Buffer.concat(chunks));
          res.end('ok');
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [saveResults()],
  resolve: {
    alias: {
      lex: at('../lex/src/index.js'),
    },
  },
  server: { fs: { allow: ['..'] } },
  optimizeDeps: { exclude: ['lex'] },
  build: { target: 'esnext', chunkSizeWarningLimit: 1200 },
});
