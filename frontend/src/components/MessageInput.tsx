import { useState, useRef, KeyboardEvent, ChangeEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Send, Square, Paperclip, Globe, X, Image as ImageIcon } from 'lucide-react'
import { uploadFiles } from '@/lib/api'
import type { FileAttachment } from '@/types'

interface Props {
  onSend: (content: string, files: FileAttachment[]) => void
  onStop: () => void
  streaming: boolean
  enableWebSearch: boolean
  forceWebSearch: boolean
  onToggleWebSearch: (v: boolean) => void
  onToggleForceWeb: (v: boolean) => void
  modelSupportsVision: boolean
  modelSupportsTools: boolean
}

export function MessageInput({
  onSend, onStop, streaming,
  enableWebSearch, forceWebSearch,
  onToggleWebSearch, onToggleForceWeb,
  modelSupportsVision, modelSupportsTools,
}: Props) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<FileAttachment[]>([])
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleSend() {
    const content = text.trim()
    if (!content && attachments.length === 0) return
    if (streaming) return
    onSend(content, attachments)
    setText('')
    setAttachments([])
  }

  async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    setUploading(true)
    try {
      const results = await uploadFiles(files)
      setAttachments(prev => [...prev, ...results])
    } catch (err) {
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  function removeAttachment(idx: number) {
    setAttachments(prev => prev.filter((_, i) => i !== idx))
  }

  const acceptedTypes = [
    '.txt', '.py', '.js', '.ts', '.html', '.css', '.md', '.json', '.yaml', '.yml',
    '.xml', '.sh', '.sql', '.csv', '.pdf', '.docx', '.xlsx', '.xls',
    ...(modelSupportsVision ? ['.png', '.jpg', '.jpeg', '.webp', '.gif'] : []),
  ].join(',')

  return (
    <div className="border-t bg-background px-4 py-3 space-y-2">
      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {attachments.map((f, i) => (
            <div key={i} className="flex items-center gap-1.5 bg-muted rounded-md px-2 py-1 text-xs">
              {f.type === 'image'
                ? <ImageIcon className="h-3.5 w-3.5 text-muted-foreground" />
                : <Paperclip className="h-3.5 w-3.5 text-muted-foreground" />
              }
              <span className="max-w-[140px] truncate">{f.name}</span>
              {f.error && <span className="text-destructive">!</span>}
              <button onClick={() => removeAttachment(i)} className="text-muted-foreground hover:text-foreground">
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Text area + actions */}
      <div className="flex gap-2 items-end">
        <Textarea
          placeholder="Mensagem... (Enter = enviar, Shift+Enter = nova linha)"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={streaming}
          rows={1}
          className="min-h-[44px] max-h-40 resize-none flex-1"
        />
        <div className="flex gap-1">
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => fileRef.current?.click()}
            disabled={streaming || uploading}
            title="Anexar arquivo"
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <input ref={fileRef} type="file" multiple hidden accept={acceptedTypes} onChange={handleFileChange} />

          {streaming ? (
            <Button type="button" variant="destructive" size="icon" onClick={onStop} title="Parar">
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button type="button" size="icon" onClick={handleSend} disabled={!text.trim() && attachments.length === 0} title="Enviar">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Web search toggles */}
      {modelSupportsTools && (
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <Switch id="ws-auto" checked={enableWebSearch} onCheckedChange={onToggleWebSearch} />
            <Label htmlFor="ws-auto" className="text-xs cursor-pointer flex items-center gap-1">
              <Globe className="h-3.5 w-3.5" /> Busca web automática
            </Label>
          </div>
          <div className="flex items-center gap-1.5">
            <Switch id="ws-force" checked={forceWebSearch} onCheckedChange={onToggleForceWeb} />
            <Label htmlFor="ws-force" className="text-xs cursor-pointer">Forçar busca</Label>
          </div>
        </div>
      )}
    </div>
  )
}
