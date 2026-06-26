import { useState, useEffect } from 'react'
import { auth } from '@/lib/api'
import { Login } from '@/pages/Login'
import { Sidebar } from '@/components/Sidebar'
import { ChatWindow } from '@/components/ChatWindow'
import type { Conversation } from '@/types'

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [currentConvId, setCurrentConvId] = useState<string | null>(null)

  useEffect(() => {
    auth.me().then(r => setAuthenticated(r.authenticated))
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

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

  function handleConvSelect(conv: Conversation) {
    setCurrentConvId(conv.id)
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar
        currentId={currentConvId}
        onSelect={handleConvSelect}
        onNew={() => setCurrentConvId(null)}
      />
      <ChatWindow
        currentConvId={currentConvId}
        onConvSaved={setCurrentConvId}
        onNewConv={() => setCurrentConvId(null)}
        onLogout={() => setAuthenticated(false)}
        darkMode={darkMode}
        onToggleDark={() => setDarkMode(d => !d)}
      />
    </div>
  )
}
