import { useState, useEffect, useRef, useCallback } from 'react'
import { auth } from '@/lib/api'
import { Login } from '@/pages/Login'
import { Sidebar } from '@/components/Sidebar'
import { ChatWindow } from '@/components/ChatWindow'

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [currentConvId, setCurrentConvId] = useState<string | null>(null)
  const sidebarRefresh = useRef<(() => void) | null>(null)

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
      <Sidebar
        currentId={currentConvId}
        onSelect={conv => setCurrentConvId(conv.id)}
        onNew={() => setCurrentConvId(null)}
        onRegisterRefresh={fn => { sidebarRefresh.current = fn }}
      />
      <ChatWindow
        currentConvId={currentConvId}
        onConvSaved={handleConvSaved}
        onNewConv={() => setCurrentConvId(null)}
        onLogout={() => setAuthenticated(false)}
        darkMode={darkMode}
        onToggleDark={() => setDarkMode(d => !d)}
      />
    </div>
  )
}
