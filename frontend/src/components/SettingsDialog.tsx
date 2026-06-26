import { useEffect, useState } from 'react'
import { settingsApi } from '@/lib/api'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Settings, CheckCircle2, CircleDashed, Loader2 } from 'lucide-react'
import type { AppSettings } from '@/types'

const KEY_LABELS: Record<string, string> = {
  openai_api_key: 'OpenAI API Key',
  anthropic_api_key: 'Anthropic API Key',
  gemini_api_key: 'Google Gemini API Key',
  groq_api_key: 'Groq API Key',
  openrouter_api_key: 'OpenRouter API Key',
  deepseek_api_key: 'DeepSeek API Key',
  ollama_base_url: 'Ollama Base URL',
  ollama_api_key: 'Ollama Cloud API Key',
  tavily_api_key: 'Tavily API Key (busca web)',
  firecrawl_api_key: 'Firecrawl API Key (scraping)',
  claude_code_oauth_token: 'Claude Code OAuth Token (assinatura)',
}

export function SettingsDialog() {
  const [open, setOpen] = useState(false)
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [newPassword, setNewPassword] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (open) {
      settingsApi.get().then(s => {
        setAppSettings(s)
        setDraft({ ollama_base_url: s.ollama_base_url || 'http://localhost:11434' })
      }).catch(() => {})
    }
  }, [open])

  async function handleSave() {
    setSaving(true)
    try {
      const payload = { ...draft }
      if (newPassword) {
        await settingsApi.changePassword(newPassword)
        setNewPassword('')
      }
      if (Object.keys(payload).length > 0) {
        await settingsApi.updateKeys(payload)
      }
      const updated = await settingsApi.get()
      setAppSettings(updated)
      setDraft({ ollama_base_url: updated.ollama_base_url || '' })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error(e)
    } finally {
      setSaving(false)
    }
  }

  const boolKeys = Object.keys(KEY_LABELS).filter(k => k !== 'ollama_base_url')

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" title="Configurações">
          <Settings className="h-5 w-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Configurações</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 pt-2">
          {/* API Keys */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Chaves de API</h3>
            <p className="text-xs text-muted-foreground">
              Deixe em branco para usar o valor do .env. Preencha para sobrescrever.
            </p>

            {/* Ollama URL */}
            <div className="space-y-1">
              <Label className="text-xs">{KEY_LABELS.ollama_base_url}</Label>
              <Input
                type="text"
                placeholder="http://localhost:11434"
                value={draft.ollama_base_url ?? ''}
                onChange={e => setDraft(d => ({ ...d, ollama_base_url: e.target.value }))}
              />
            </div>

            {boolKeys.map(key => (
              <div key={key} className="space-y-1">
                <Label className="text-xs flex items-center gap-1.5">
                  {appSettings?.[key as keyof AppSettings]
                    ? <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                    : <CircleDashed className="h-3.5 w-3.5 text-muted-foreground" />
                  }
                  {KEY_LABELS[key]}
                </Label>
                <Input
                  type="password"
                  placeholder={appSettings?.[key as keyof AppSettings] ? '●●●●●●● (configurado)' : 'Cole a chave aqui'}
                  value={draft[key] ?? ''}
                  onChange={e => setDraft(d => ({ ...d, [key]: e.target.value }))}
                />
              </div>
            ))}
          </div>

          <Separator />

          {/* Password */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">Alterar senha</h3>
            <Input
              type="password"
              placeholder="Nova senha"
              value={newPassword}
              onChange={e => setNewPassword(e.target.value)}
            />
          </div>

          <Button onClick={handleSave} disabled={saving} className="w-full">
            {saving
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : saved ? '✓ Salvo!' : 'Salvar'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
