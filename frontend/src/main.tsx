import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import AdminApp from './AdminApp.tsx'
import FloatingWidgetApp from './FloatingWidgetApp.tsx'

const params = new URLSearchParams(window.location.search)
const uiMode = params.get('ui')
const isAdminRoute = window.location.pathname.startsWith("/admin")
const rootComponent = uiMode === 'floating' ? FloatingWidgetApp : isAdminRoute ? AdminApp : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {rootComponent === FloatingWidgetApp ? <FloatingWidgetApp /> : rootComponent === AdminApp ? <AdminApp /> : <App />}
  </StrictMode>,
)
