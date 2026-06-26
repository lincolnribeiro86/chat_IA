import { useState, useCallback, useEffect, useRef } from 'react'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { ModelSelector } from './ModelSelector'
import { SettingsDialog } from './SettingsDialog'
import { useChat } from '@/hooks/useChat'
import { conversationsApi, auth } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { LogOut, Moon, Sun, Sliders } from 'lucide-react'
import type { ModelInfo, FileAttachment } from '@/types'

interface Props {
  currentConvId: string | null
  onConvSaved: (id: string) => void
  onNewConv: () => void
  onLogout: () => void
  darkMode: boolean
  onToggleDark: () => void
}

export function ChatWindow({
  currentConvId, onConvSaved, onNewConv, onLogout, darkMode, onToggleDark
}: Props) {
  const [selectedModel, setSelectedModel] = useState<ModelInfo>({
    id: 'llama3.2',
    name: 'Llama 3.2',
    provider: 'ollama',
    supports_vision: true,
    supports_tools: true,
    context_window: 8192,
  })
  const [temperature, setTemperature] = useState(0.5)
  const [enableWebSearch, setEnableWebSearch] = useState(false)
  const [forceWebSearch, setForceWebSearch] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const { messages, streaming, append, stop, clear, loadMessages } = useChat({
    modelId: selectedModel.id,
    temperature,
    enableWebSearch,
    forceWebSearch,
  })

  // Limpa o chat quando o sidebar clica em "Nova conversa" (currentConvId → null)
  const prevConvId = useRef(currentConvId)
  useEffect(() => {
    if (prevConvId.current !== null && currentConvId === null) {
      clear()
    }
    prevConvId.current = currentConvId
  }, [currentConvId, clear])

  // Carrega a conversa quando selecionada no sidebar
  useEffect(() => {
    if (currentConvId) {
      conversationsApi.load(currentConvId).then(data => {
        const msgs = (data.messages as Array<{ role: string; content: string }>).map((m, i) => ({
          id: `loaded-${i}`,
          role: m.role as 'user' | 'assistant',
          content: m.content,
        }))
        loadMessages(msgs)
      }).catch(() => {})
    }
  }, [currentConvId, loadMessages])

  const handleSend = useCallback(async (content: string, files: FileAttachment[]) => {
    await append(content, files)

    // Auto-save conversation
    const title = content.slice(0, 80) + (content.length > 80 ? '...' : '') || 'Conversa'
    if (!currentConvId) {
      try {
        const { id } = await conversationsApi.save(
          [...messages, { id: '', role: 'user' as const, content }],
          selectedModel.id,
          title,
        )
        onConvSaved(id)
      } catch { /* DB not available */ }
    } else {
      try {
        await conversationsApi.update(currentConvId, messages, selectedModel.id)
      } catch { /* ignore */ }
    }
  }, [append, messages, currentConvId, selectedModel.id, onConvSaved])

  const handleNew = useCallback(() => {
    clear()
    onNewConv()
  }, [clear, onNewConv])

  const handleLogout = async () => {
    await auth.logout()
    onLogout()
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Top bar */}
      <header className="flex items-center gap-2 px-4 py-2 border-b bg-background shrink-0">
        <div className="flex-1">
          <ModelSelector selectedId={selectedModel.id} onSelect={setSelectedModel} />
        </div>

        {/* Temperature slider */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Sliders className="h-3.5 w-3.5" />
          <input
            type="range" min="0" max="1" step="0.05"
            value={temperature}
            onChange={e => setTemperature(Number(e.target.value))}
            className="w-20 accent-primary"
            title={`Temperatura: ${temperature}`}
          />
          <span className="w-6">{temperature.toFixed(1)}</span>
        </div>

        <SettingsDialog />

        <Button variant="ghost" size="icon" onClick={onToggleDark} title="Tema">
          {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        <Button variant="ghost" size="icon" onClick={handleLogout} title="Sair">
          <LogOut className="h-4 w-4" />
        </Button>
      </header>

      {/* Messages */}
      <MessageList messages={messages} />

      {/* Input */}
      <MessageInput
        onSend={handleSend}
        onStop={stop}
        streaming={streaming}
        enableWebSearch={enableWebSearch}
        forceWebSearch={forceWebSearch}
        onToggleWebSearch={setEnableWebSearch}
        onToggleForceWeb={setForceWebSearch}
        modelSupportsVision={selectedModel.supports_vision}
        modelSupportsTools={selectedModel.supports_tools}
      />
    </div>
  )
}
