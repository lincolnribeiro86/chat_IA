import { useEffect } from 'react'
import { conversationsApi } from '@/lib/api'
import { useConversations } from '@/hooks/useConversations'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { MessageSquare, Plus, Trash2, Download, Bot } from 'lucide-react'
import type { Conversation } from '@/types'

interface Props {
  currentId: string | null
  onSelect: (conv: Conversation) => void
  onNew: () => void
  onRegisterRefresh?: (fn: () => void) => void
  width?: number
}

export function Sidebar({ currentId, onSelect, onNew, onRegisterRefresh }: Props) {
  const { conversations, loading, refresh, remove, exportConv } = useConversations()

  useEffect(() => {
    refresh()
    onRegisterRefresh?.(refresh)
  }, [refresh, onRegisterRefresh])

  return (
    <aside className="w-full border-r bg-card flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b shrink-0">
        <Bot className="h-5 w-5 text-primary shrink-0" />
        <span className="font-semibold text-sm truncate">chat_IA</span>
      </div>

      {/* New conversation */}
      <div className="px-3 pt-3 pb-2 shrink-0">
        <Button onClick={onNew} variant="outline" size="sm" className="w-full gap-2">
          <Plus className="h-4 w-4" /> Nova conversa
        </Button>
      </div>

      <Separator />

      {/* Conversation list */}
      <ScrollArea className="flex-1">
        <div className="px-2 py-2 space-y-0.5">
          {loading && <p className="text-xs text-muted-foreground px-2 py-1">Carregando...</p>}
          {!loading && conversations.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-1">Nenhuma conversa salva</p>
          )}
          {conversations.map(conv => (
            <ConvItem
              key={conv.id}
              conv={conv}
              active={conv.id === currentId}
              onSelect={() => onSelect(conv)}
              onDelete={() => remove(conv.id)}
              onExportMd={() => exportConv(conv.id, 'markdown')}
              onExportJson={() => exportConv(conv.id, 'json')}
            />
          ))}
        </div>
      </ScrollArea>
    </aside>
  )
}

function ConvItem({
  conv, active, onSelect, onDelete, onExportMd, onExportJson,
}: {
  conv: Conversation
  active: boolean
  onSelect: () => void
  onDelete: () => void
  onExportMd: () => void
  onExportJson: () => void
}) {
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(`Excluir "${conv.title}"?`)) onDelete()
  }

  return (
    <div
      className={`group flex items-center rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors ${active ? 'bg-accent' : ''}`}
      onClick={onSelect}
    >
      <MessageSquare className="h-3.5 w-3.5 mr-2 text-muted-foreground shrink-0" />
      <span className="flex-1 text-xs truncate">{conv.title}</span>

      {/* Action buttons — always visible on active, hover on others */}
      <div className={`flex items-center gap-0.5 ml-1 ${active ? 'flex' : 'hidden group-hover:flex'}`}>
        <button
          onClick={e => { e.stopPropagation(); onExportMd() }}
          className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
          title="Exportar Markdown"
        >
          <Download className="h-3 w-3" />
        </button>
        <button
          onClick={handleDelete}
          className="p-1 rounded hover:bg-background text-muted-foreground hover:text-destructive"
          title="Excluir conversa"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
    </div>
  )
}
