import { Badge } from '@/components/ui/badge'
import { formatCost, formatTokens } from '@/lib/utils'
import type { UsageInfo } from '@/types'

interface Props { usage: UsageInfo }

export function CostBadge({ usage }: Props) {
  return (
    <div className="flex items-center gap-1.5 mt-1">
      <Badge variant="outline" className="text-xs text-muted-foreground font-normal px-1.5 py-0">
        ↑ {formatTokens(usage.input_tokens)} ↓ {formatTokens(usage.output_tokens)} tokens
      </Badge>
      {usage.cost_usd != null && (
        <Badge variant="outline" className="text-xs text-muted-foreground font-normal px-1.5 py-0">
          {formatCost(usage.cost_usd)}
        </Badge>
      )}
    </div>
  )
}
