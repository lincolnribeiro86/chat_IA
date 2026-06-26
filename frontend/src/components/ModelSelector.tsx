import { useEffect, useState, useCallback } from 'react'
import { modelsApi } from '@/lib/api'
import type { ModelInfo, ProviderGroup } from '@/types'
import { ChevronDown, Cpu, Star } from 'lucide-react'

const STORAGE_KEY = 'chatia_fav_models'

function loadFavs(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? new Set(JSON.parse(raw)) : new Set()
  } catch {
    return new Set()
  }
}

function saveFavs(favs: Set<string>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...favs]))
}

interface Props {
  selectedId: string
  onSelect: (model: ModelInfo) => void
}

export function ModelSelector({ selectedId, onSelect }: Props) {
  const [providers, setProviders] = useState<ProviderGroup[]>([])
  const [open, setOpen] = useState(false)
  const [favs, setFavs] = useState<Set<string>>(loadFavs)

  useEffect(() => {
    modelsApi.list().then(r => setProviders(r.providers)).catch(() => {})
  }, [])

  const toggleFav = useCallback((e: React.MouseEvent, modelId: string) => {
    e.stopPropagation()
    setFavs(prev => {
      const next = new Set(prev)
      next.has(modelId) ? next.delete(modelId) : next.add(modelId)
      saveFavs(next)
      return next
    })
  }, [])

  const allModels = providers.flatMap(p => p.models)
  const selectedModel = allModels.find(m => m.id === selectedId)
  const favModels = allModels.filter(m => favs.has(m.id))

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 text-sm px-3 py-1.5 rounded-md border bg-background hover:bg-accent transition-colors w-full"
      >
        <Cpu className="h-4 w-4 text-muted-foreground shrink-0" />
        <span className="flex-1 text-left truncate flex items-center gap-1.5">
          {favs.has(selectedId) && <Star className="h-3 w-3 fill-yellow-400 text-yellow-400 shrink-0" />}
          {selectedModel?.name ?? 'Selecionar modelo...'}
        </span>
        <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute top-full left-0 mt-1 w-80 max-h-[28rem] overflow-y-auto z-20 rounded-md border bg-background shadow-lg">

            {/* Seção de favoritos */}
            {favModels.length > 0 && (
              <div>
                <div className="px-3 py-1.5 text-xs font-semibold text-yellow-500 sticky top-0 bg-background border-b flex items-center gap-1">
                  <Star className="h-3 w-3 fill-yellow-400" /> Favoritos
                </div>
                {favModels.map(model => (
                  <ModelRow
                    key={`fav-${model.id}`}
                    model={model}
                    selectedId={selectedId}
                    isFav={true}
                    onSelect={() => { onSelect(model); setOpen(false) }}
                    onToggleFav={toggleFav}
                  />
                ))}
              </div>
            )}

            {/* Grupos por provedor */}
            {providers.map(prov => (
              <div key={prov.id}>
                <div className="px-3 py-1.5 text-xs font-semibold text-muted-foreground sticky top-0 bg-background border-b">
                  {prov.name}
                </div>
                {prov.models.map(model => (
                  <ModelRow
                    key={model.id}
                    model={model}
                    selectedId={selectedId}
                    isFav={favs.has(model.id)}
                    onSelect={() => { onSelect(model); setOpen(false) }}
                    onToggleFav={toggleFav}
                  />
                ))}
              </div>
            ))}

          </div>
        </>
      )}
    </div>
  )
}

interface RowProps {
  model: ModelInfo
  selectedId: string
  isFav: boolean
  onSelect: () => void
  onToggleFav: (e: React.MouseEvent, id: string) => void
}

function ModelRow({ model, selectedId, isFav, onSelect, onToggleFav }: RowProps) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2 text-sm hover:bg-accent transition-colors flex items-center gap-2 group ${model.id === selectedId ? 'bg-accent' : ''}`}
    >
      {/* Estrela */}
      <span
        role="button"
        onClick={e => onToggleFav(e, model.id)}
        className={`shrink-0 transition-colors ${isFav ? 'text-yellow-400' : 'text-muted-foreground/30 group-hover:text-muted-foreground/60'}`}
        title={isFav ? 'Remover dos favoritos' : 'Adicionar aos favoritos'}
      >
        <Star className={`h-3.5 w-3.5 ${isFav ? 'fill-yellow-400' : ''}`} />
      </span>

      <span className="truncate flex-1">{model.name}</span>

      {/* Badges */}
      <div className="flex gap-1 shrink-0 items-center">
        {model.usage_tier && <UsageTierBadge tier={model.usage_tier} />}
        {model.supports_vision && (
          <span className="text-[10px] bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-1 rounded">👁</span>
        )}
        {model.supports_tools && (
          <span className="text-[10px] bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-1 rounded">🔧</span>
        )}
      </div>
    </button>
  )
}

function UsageTierBadge({ tier }: { tier: string }) {
  const cfg: Record<string, { label: string; cls: string }> = {
    low:        { label: 'Free',    cls: 'bg-emerald-100 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300' },
    medium:     { label: 'Med',     cls: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300' },
    high:       { label: 'High',    cls: 'bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300' },
    extra_high: { label: 'Pro',     cls: 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' },
  }
  const { label, cls } = cfg[tier] ?? { label: tier, cls: 'bg-muted text-muted-foreground' }
  return <span className={`text-[10px] px-1 rounded font-medium ${cls}`}>{label}</span>
}
