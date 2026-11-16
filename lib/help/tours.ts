/**
 * Tour registry.
 *
 * Tours are manual: nothing here opens itself, and no first-visit state is
 * persisted. A route with an entry gets a "Take a tour" button and a row in the
 * help popover; a route without one gets neither.
 */
export type TourPosition = 'top' | 'bottom' | 'left' | 'right';

export interface TourStep {
  /** CSS selector for the element this step points at. */
  target: string;
  title: string;
  description: string;
  /** Preferred side of the target. Falls back to another side when there is
   *  not enough room. Defaults to `bottom`. */
  position?: TourPosition;
}

export interface HelpTour {
  id: string;
  /** Route this tour belongs to, without query string or trailing slash. */
  route: string;
  title: string;
  steps: TourStep[];
}

export const HELP_TOURS: HelpTour[] = [
  {
    id: 'predict-performance',
    route: '/predict',
    title: 'Predict performance',
    steps: [
      {
        target: '[data-tour="model"]',
        title: 'Start with a model',
        description:
          'Type a Hugging Face model ID or choose from the list. The list only shows models the selected simulator can price.',
        position: 'bottom',
      },
      {
        target: '[data-tour="warning"]',
        title: 'Check the assumptions',
        description:
          'This strip shows the workload the estimate will use: sequence lengths, KV cache precision, and concurrency. Use Adjust to change them.',
        position: 'bottom',
      },
      {
        target: '[data-tour="calculate"]',
        title: 'Calculate when you are ready',
        description:
          'Edit any input at any time. Results are not refreshed until you press Calculate, so what you see always matches the last submitted configuration.',
        position: 'bottom',
      },
      {
        target: '[data-tour="result-tile-gpus"]',
        title: 'Your results at a glance',
        description:
          'These tiles show GPU count, memory requirements, and monthly cost. Click any tile to see the math behind it.',
        position: 'right',
      },
      {
        target: '[data-tour="assumptions"]',
        title: 'Fine-tune your workload',
        description:
          'Expand a section to adjust traffic patterns, sequence lengths, and hardware settings. Press Calculate again after changing a value.',
        position: 'top',
      },
    ],
  },
  {
    id: 'recommend-sizing',
    route: '/recommend',
    title: 'Recommend sizing',
    steps: [
      {
        target: '[data-tour="model"]',
        title: 'Start from a model and GPU',
        description:
          'Choose a model and a GPU system. These two choices decide which configurations the service can search.',
        position: 'bottom',
      },
      {
        target: '[data-tour="goal"]',
        title: 'Choose a sizing goal',
        description:
          'Latency SLO meets TTFT/TPOT targets, throughput demand sustains an arrival rate, and concurrency capacity holds a number of resident sequences. They can return different GPU counts.',
        position: 'bottom',
      },
      {
        target: '[data-tour="targets"]',
        title: 'Check the workload',
        description:
          'This strip states the sequence lengths, latency bounds, and concurrency the search will use. Use Adjust to edit them below.',
        position: 'bottom',
      },
      {
        target: '[data-tour="calculate"]',
        title: 'Calculate, or map a frontier',
        description:
          'Calculate returns the smallest configuration that meets the goal. Pareto frontier lists the trade-offs instead of one answer.',
        position: 'bottom',
      },
      {
        target: '[data-tour="results"]',
        title: 'Read the recommendation',
        description:
          'The tiles summarise GPUs, topology, and expected performance. Click a tile to see how the number was derived.',
        position: 'right',
      },
    ],
  },
  {
    id: 'kv-cache',
    route: '/kv-cache',
    title: 'KV cache calculator',
    steps: [
      {
        target: '[data-tour="model"]',
        title: 'Pick the model and GPU',
        description:
          'Memory depends on both: the model sets per-token KV size, and the GPU sets capacity and how much of it the backend may use.',
        position: 'bottom',
      },
      {
        target: '[data-tour="workload"]',
        title: 'Check the workload',
        description:
          'The strip shows the maximum tokens and batch size the estimate will use. Use Adjust to change backend, version, or parallelism.',
        position: 'bottom',
      },
      {
        target: '[data-tour="calculate"]',
        title: 'Calculate memory',
        description:
          'Press Calculate after any change. Memory is estimated by the service, so the browser never derives these numbers itself.',
        position: 'bottom',
      },
      {
        target: '[data-tour="results"]',
        title: 'Read the breakdown',
        description:
          'Weights, KV cache, and free headroom are reported separately. A configuration that fits still needs headroom for the batch you expect.',
        position: 'top',
      },
    ],
  },
  {
    id: 'gpu-explorer',
    route: '/gpu-explorer',
    title: 'GPU explorer',
    steps: [
      {
        target: '[data-tour="presets"]',
        title: 'Start from a preset',
        description:
          'A preset picks a useful pair of axes. Cost efficiency appears only when hardware pricing is enabled and loaded.',
        position: 'bottom',
      },
      {
        target: '[data-tour="vendor"]',
        title: 'Filter by vendor',
        description:
          'The filter applies to the chart and to any cost coverage check, so an empty vendor is empty for a reason.',
        position: 'bottom',
      },
      {
        target: '[data-tour="axes"]',
        title: 'Choose what to compare',
        description:
          'Each axis is a hardware property. Changing an axis does not run a model; it re-plots the same catalog.',
        position: 'bottom',
      },
      {
        target: '[data-tour="chart"]',
        title: 'Read the chart',
        description:
          'Position shows the two chosen axes. Up and to the right means more memory and more throughput, not a better fit for every model.',
        position: 'top',
      },
      {
        target: '[data-tour="legend"]',
        title: 'Know what bubble size means',
        description:
          'Bubble size is either BF16 TFLOPs or cost efficiency, depending on pricing availability. The throughput index is a planning metric, not measured throughput.',
        position: 'top',
      },
    ],
  },
  {
    id: 'hybrid-savings',
    route: '/hybrid-savings',
    title: 'Hybrid savings',
    steps: [
      {
        target: '[data-tour="workload"]',
        title: 'Describe the workload',
        description:
          'Choose a model and monthly token volumes, or start from a workload preset. Advanced workload holds the request shape and performance targets.',
        position: 'bottom',
      },
      {
        target: '[data-tour="assumptions"]',
        title: 'Check the planning assumptions',
        description:
          'This strip summarises request size, processing hours, and billing mode. Cost assumptions such as hardware life and power are edited on the Sources page.',
        position: 'bottom',
      },
      {
        target: '[data-tour="calculate"]',
        title: 'Calculate the comparison',
        description:
          'The service sizes the hardware first. Pricing is then applied to the sized topology, so a price change never changes a GPU count.',
        position: 'bottom',
      },
      {
        target: '[data-tour="results"]',
        title: 'Compare the three cost paths',
        description:
          'Hosted, rented, and purchased are compared per million output tokens. Cost view switches between full TCO and direct running costs; the cheapest card is an output, not an assumption.',
        position: 'top',
      },
    ],
  },
  {
    id: 'validation-compare',
    route: '/validation/compare',
    title: 'Benchmark comparison',
    steps: [
      {
        target: '[data-tour="inputs"]',
        title: 'Pick a measured run',
        description:
          'Start with the model and GPU, then choose sequence lengths and tensor parallel size. The measured runtime configuration keeps results from different runs separate.',
        position: 'bottom',
      },
      {
        target: '[data-tour="compare"]',
        title: 'Run the comparison',
        description:
          'Compare predicts the same configuration with the selected simulator and plots it beside the measured rows.',
        position: 'top',
      },
      {
        target: '[data-tour="results"]',
        title: 'Read measured against predicted',
        description:
          'Measured and predicted series share an axis. A gap shows where the estimate and the hardware disagree, not which one is wrong.',
        position: 'top',
      },
    ],
  },
];

/** Normalise a pathname for registry lookup: drop query, hash, trailing slash. */
export function normaliseRoute(pathname: string | null | undefined): string {
  if (!pathname) return '/';
  const withoutQuery = pathname.split('?')[0].split('#')[0];
  if (withoutQuery.length > 1 && withoutQuery.endsWith('/')) {
    return withoutQuery.slice(0, -1);
  }
  return withoutQuery || '/';
}

export function getTourForRoute(pathname: string | null | undefined): HelpTour | undefined {
  const route = normaliseRoute(pathname);
  return HELP_TOURS.find((tour) => tour.route === route);
}

export function getTourById(id: string): HelpTour | undefined {
  return HELP_TOURS.find((tour) => tour.id === id);
}
