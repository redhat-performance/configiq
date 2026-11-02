import { redirect } from 'next/navigation'
import { legacyPairRedirectTarget } from '@/lib/tested-models/links'
import { TestedModelsIndex } from './TestedModelsIndex'

export const dynamic = 'force-dynamic'

interface TestedModelsPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>
}

function single(value: string | string[] | undefined): string {
  return typeof value === 'string' ? value : ''
}

export default async function TestedModelsPage({ searchParams }: TestedModelsPageProps) {
  const params = await searchParams
  const legacyPair = single(params.pair)
  if (legacyPair) redirect(legacyPairRedirectTarget(legacyPair))

  return (
    <TestedModelsIndex
      initialModel={single(params.model)}
      initialSystem={single(params.system)}
    />
  )
}
