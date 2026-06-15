import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  envDir: '..',
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/mcp': 'http://127.0.0.1:8000',
      '/docs': 'http://127.0.0.1:8000',
      '/yuque': 'http://127.0.0.1:8000',
      '/admin-api': 'http://127.0.0.1:8000',
      '/admin-media': 'http://127.0.0.1:8000',
      '/visitor': 'http://127.0.0.1:8000',
    },
  },
})
