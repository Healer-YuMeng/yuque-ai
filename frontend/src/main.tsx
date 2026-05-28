import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import FloatingWidgetApp from './FloatingWidgetApp.tsx'

const params = new URLSearchParams(window.location.search)
const uiMode = params.get('ui')
const rootComponent = uiMode === 'floating' ? FloatingWidgetApp : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {rootComponent === FloatingWidgetApp ? <FloatingWidgetApp /> : <App />}
  </StrictMode>,
)
