import { useState, useEffect, useRef, useCallback } from 'react'
import { auth } from '@/lib/api'
import { Login } from '@/pages/Login'
import { Sidebar } from '@/components/Sidebar'
import { ChatWindow } from '@/components/ChatWindow'
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react'

const SIDEBAR_MIN = 180
const SIDEBAR_MAX = 480
const SIDEBAR_DEFAULT = 256

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [currentConvId, setCurrentConvId] = useState<string | null>(null)
  const sidebarRefresh = useRef<(() => void) | null>(null)

  // Sidebar width + collapsed state
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem('sidebar_width')
    return saved ? Number(saved) : SIDEBAR_DEFAULT
  })
  const [collapsed, setCollapsed] = useState(() =>
    localStorage.getItem('sidebar_collapsed') === 'true'
  )
  const dragging = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true
    startX.current = e.clientX
    startW.current = sidebarWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [sidebarWidth])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return
      const delta = e.clientX - startX.current
      const next = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startW.current + delta))
      setSidebarWidth(next)
      if (next <= SIDEBAR_MIN + 20) setCollapsed(true)
      else setCollapsed(false)
    }
    const onUp = () => {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      setSidebarWidth(w => {
        localStorage.setItem('sidebar_width', String(w))
        return w
      })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  const toggleCollapsed = useCallback(() => {
    setCollapsed(c => {
      const next = !c
      localStorage.setItem('sidebar_collapsed', String(next))
      return next
    })
  }, [])

  useEffect(() => {
    auth.me().then(r => setAuthenticated(r.authenticated))
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  const handleConvSaved = useCallback((id: string) => {
    setCurrentConvId(id)
    sidebarRefresh.current?.()
  }, [])

  if (authenticated === null) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!authenticated) {
    return <Login onLogin={() => setAuthenticated(true)} />
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <div
        className="relative flex-shrink-0 transition-[width] duration-0"
        style={{ width: collapsed ? 0 : sidebarWidth }}
      >
        <div className="absolute inset-0 overflow-hidden">
          <Sidebar
            currentId={currentConvId}
            onSelect={conv => setCurrentConvId(conv.id)}
            onNew={() => setCurrentConvId(null)}
            onRegisterRefresh={fn => { sidebarRefresh.current = fn }}
            width={sidebarWidth}
          />
        </div>

        {/* Drag handle */}
        {!collapsed && (
          <div
            onMouseDown={onMouseDown}
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-primary/40 active:bg-primary/60 z-30 transition-colors"
            title="Arrastar para redimensionar"
          />
        )}
      </div>

      {/* Toggle button */}
      <button
        onClick={toggleCollapsed}
        className="flex-shrink-0 w-5 flex items-center justify-center bg-background border-r hover:bg-accent transition-colors z-20 text-muted-foreground hover:text-foreground"
        title={collapsed ? 'Abrir sidebar' : 'Fechar sidebar'}
      >
        {collapsed
          ? <PanelLeftOpen className="h-3.5 w-3.5" />
          : <PanelLeftClose className="h-3.5 w-3.5" />
        }
      </button>

      {/* Chat */}
      <div className="flex-1 min-w-0">
        <ChatWindow
          currentConvId={currentConvId}
          onConvSaved={handleConvSaved}
          onNewConv={() => setCurrentConvId(null)}
          onLogout={() => setAuthenticated(false)}
          darkMode={darkMode}
          onToggleDark={() => setDarkMode(d => !d)}
        />
      </div>
    </div>
  )
}
