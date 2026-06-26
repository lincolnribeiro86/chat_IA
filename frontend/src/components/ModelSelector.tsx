import { useEffect, useState } from 'react'
import { modelsApi } from '@/lib/api'
import type { ModelInfo, ProviderGroup } from '@/types'
import { ChevronDown, Cpu } from 'lucide-react'

interface Props {
  selectedId: string
  onSelect: (model: ModelInfo) => void
}

export function ModelSelector({ selectedId, onSelect }: Props) {
  const [providers, setProviders] = useState<ProviderGroup[]>([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    modelsApi.list().then(r => setProviders(r.providers)).catch(() => {})
  }, [])

  const selectedModel = providers.flatMap(p => p.models).find(m => m.id === selectedId)

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-md border bg-background hover:bg-accent transition-colors w-full"
      >
        <Cpu className="h-4 w-4 text-muted-foreground shrink-0" />
        <span className="flex-1 text-left truncate">
          {selectedModel?.name ?? 'Selecionar modelo...'}
        </span>
        <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute top-full left-0 mt-1 w-72 max-h-96 overflow-y-auto z-20 rounded-md border bg-background shadow-lg">
            {providers.map(prov => (
              <div key={prov.id}>
                <div className="px-3 py-1.5 text-xs font-semibold text-muted-foreground sticky top-0 bg-background border-b">
                  {prov.name}
                </div>
                {prov.models.map(model => (
                  <button
                    key={model.id}
                    onClick={() => { onSelect(model); setOpen(false) }}
                    className={`w-full text-left px-3 py-2 text-sm hover:bg-accent transition-colors flex items-center justify-between gap-2 ${model.id === selectedId ? 'bg-accent' : ''}`}
                  >
                    <span className="truncate">{model.name}</span>
                    <div className="flex gap-1 shrink-0">
                      {model.supports_vision && (
                        <span className="text-[10px] bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-1 rounded">👁</span>
                      )}
                      {model.supports_tools && (
                        <span className="text-[10px] bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-1 rounded">🔧</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
