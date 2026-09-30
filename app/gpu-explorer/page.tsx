"use client";

// Cost-related axes (hardware cost) and the cost-efficiency bubble-size metric are
// driven by the costings API via useCostings, gated on the costingsEnabled setting.
// Cloud pricing per provider is applied on the Sources page selection.

import * as React from 'react';
import {
  PageSection,
  Title,
  Content as TextContent,
  ToggleGroup,
  ToggleGroupItem,
  Card,
  CardBody,
  Flex,
  FlexItem,
  Label,
  Content as Text
} from "@patternfly/react-core";
import { useCatalog, type GpuOption } from '@/lib/hooks/useCatalog';
import { useSettings } from '@/contexts/SettingsContext';
import { useCostings } from '@/lib/hooks/useCostings';
import { GpuBubbleChart } from './GpuBubbleChart';
import styles from './gpu-explorer.module.css';

type Preset = 'balanced' | 'cost-efficiency' | 'performance';
type XAxis = 'vram' | 'price' | 'throughput-index' | 'mem-bw';
type YAxis = 'vram' | 'price' | 'throughput-index' | 'mem-bw';

const ARCH_MULTIPLIER: Record<string, number> = {
  'blackwell':    1.3,
  'hopper':       1.2,
  'ada-lovelace': 1.0,
  'ampere':       0.85,
}

export default function GpuExplorerPage() {
  const [mounted, setMounted] = React.useState(false);
  const [preset, setPreset] = React.useState<Preset>('balanced');
  const [xAxis, setXAxis] = React.useState<XAxis>('vram');
  const [yAxis, setYAxis] = React.useState<YAxis>('throughput-index');
  const [vendorFilter, setVendorFilter] = React.useState<'all' | 'nvidia' | 'amd'>('all');
  const { gpuOptions, isLoading } = useCatalog();
  const { costingsEnabled, preferredCloudProvider, pricingSource } = useSettings();
  const costings = useCostings(costingsEnabled, pricingSource);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  const filteredGPUs = gpuOptions.filter(gpu => {
    if (vendorFilter === 'all') return true;
    return gpu.vendor === vendorFilter;
  });

  // Cost axes/preset are only meaningful once live hardware costs have loaded
  // for the GPUs actually on screen; otherwise the 'price' axis is an all-zero
  // column. Evaluate coverage over the current vendor-filtered set, not the
  // global cost map, so e.g. filtering to a vendor with no priced GPUs disables
  // the cost controls rather than plotting them all at zero.
  const hasHardwareCosts = filteredGPUs.some(
    gpu => (costings.gpuHardwareCosts.get(gpu.systemId)?.new_usd ?? 0) > 0,
  );
  const costMetricsReady = costingsEnabled && hasHardwareCosts;

  // If cost metrics become unavailable (costings off, or the displayed GPUs have
  // no hardware costs) while a price axis or the cost-efficiency preset is
  // active, fall back to safe defaults so the chart never shows an empty axis.
  React.useEffect(() => {
    if (!costMetricsReady) {
      if (preset === 'cost-efficiency') setPreset('balanced');
      setXAxis(prev => (prev === 'price' ? 'vram' : prev));
      setYAxis(prev => (prev === 'price' ? 'throughput-index' : prev));
    }
  }, [costMetricsReady, preset]);

  const calculateThroughputIndex = (gpu: GpuOption) => {
    const multiplier = ARCH_MULTIPLIER[gpu.architecture ?? ''] ?? 1.0;
    const bwGbps = (gpu.bandwidthTbps ?? 0) * 1000;
    const vram = gpu.vramGb ?? 0;
    return bwGbps * multiplier * (vram / 80);
  };

  // Relative cost-efficiency metric: throughput index per $1k of hardware cost.
  // Uses live hardware costs from the costings API; 0 when no cost is known so
  // the bubble falls back to a minimum radius rather than skewing the scale.
  const throughputPerKUsd = (gpu: GpuOption): number => {
    const hw = costings.gpuHardwareCosts.get(gpu.systemId)?.new_usd;
    if (!hw || hw <= 0) return 0;
    return calculateThroughputIndex(gpu) / (hw / 1000);
  };

  const getAxisValue = (gpu: GpuOption, axis: XAxis | YAxis): number => {
    switch (axis) {
      case 'vram':            return gpu.vramGb ?? 0;
      case 'price': {
        const hw = costings.gpuHardwareCosts.get(gpu.systemId)
        return hw?.new_usd ?? 0
      }
      case 'throughput-index': return calculateThroughputIndex(gpu);
      case 'mem-bw':          return (gpu.bandwidthTbps ?? 0) * 1000;
      default:                return 0;
    }
  };

  // Get axis label
  const getAxisLabel = (axis: XAxis | YAxis): string => {
    switch (axis) {
      case 'vram':
        return 'VRAM (GB)';
      case 'price':
        return 'Hardware Cost (USD)';
      case 'throughput-index':
        return 'Throughput Index';
      case 'mem-bw':
        return 'Memory Bandwidth (GB/s)';
      default:
        return '';
    }
  };

  // Apply preset
  React.useEffect(() => {
    switch (preset) {
      case 'balanced':
        setXAxis('vram');
        setYAxis('throughput-index');
        break;
      case 'cost-efficiency':
        setXAxis('price'); // Hardware cost
        setYAxis('throughput-index');
        break;
      case 'performance':
        setXAxis('mem-bw');
        setYAxis('throughput-index');
        break;
    }
  }, [preset]);

  // Prepare all data for bubble chart with full specs
  const allData = filteredGPUs.map(gpu => ({
    x: getAxisValue(gpu, xAxis),
    y: getAxisValue(gpu, yAxis),
    // When costings is enabled and hardware costs are loaded, bubble size shows
    // throughput per $1k of hardware cost; otherwise it falls back to BF16 TFLOPs.
    size: costMetricsReady ? throughputPerKUsd(gpu) : (gpu.tflopsBf16 ?? 0),
    name: gpu.label.replace(/NVIDIA |AMD /i, ''),
    fullName: gpu.label,
    color: gpu.vendor === 'amd' ? '#c55a5a' : '#5b9bd5',
    vram: gpu.vramGb ?? 0,
    hwCost: costings.gpuHardwareCosts.get(gpu.systemId)?.new_usd ?? undefined,
    tokensPerDollar: throughputPerKUsd(gpu),
    memBW: (gpu.bandwidthTbps ?? 0) * 1000,
    architecture: gpu.architecture
      ? gpu.architecture.charAt(0).toUpperCase() + gpu.architecture.slice(1)
      : 'Unknown',
    tflops: gpu.tflopsBf16 ?? 0,
  }));

  return (
    <>
      <div style={{ padding: '20px 24px 0' }}>
        <h1 className={styles.pageTitle}>GPU explorer</h1>
        <p className={styles.subtitle}>LLM inference planning — compare GPU generations by memory, bandwidth, and cost efficiency</p>
      </div>

      <PageSection>
        <Card>
          <CardBody>
            <Flex direction={{ default: 'column' }} spaceItems={{ default: 'spaceItemsMd' }}>
              {/* Presets */}
              <FlexItem>
                <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                  Preset:
                </Text>
                <ToggleGroup>
                  <ToggleGroupItem text="Balanced" isSelected={preset === 'balanced'} onChange={() => setPreset('balanced')} />
                  <ToggleGroupItem text="Cost efficiency" isSelected={preset === 'cost-efficiency'} onChange={() => setPreset('cost-efficiency')} isDisabled={!costMetricsReady} />
                  <ToggleGroupItem text="Performance" isSelected={preset === 'performance'} onChange={() => setPreset('performance')} />
                </ToggleGroup>
              </FlexItem>

              {/* Vendor Filter */}
              <FlexItem>
                <Flex spaceItems={{ default: 'spaceItemsLg' }}>
                  <FlexItem>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', fontSize: '14px' }}>
                      Vendor:
                    </Text>
                  </FlexItem>
                  <FlexItem>
                    <Label
                      color={vendorFilter === 'nvidia' ? 'blue' : 'grey'}
                      onClick={() => setVendorFilter(vendorFilter === 'nvidia' ? 'all' : 'nvidia')}
                      style={{ cursor: 'pointer' }}
                    >
                      NVIDIA
                    </Label>
                  </FlexItem>
                  <FlexItem>
                    <Label
                      color={vendorFilter === 'amd' ? 'red' : 'grey'}
                      onClick={() => setVendorFilter(vendorFilter === 'amd' ? 'all' : 'amd')}
                      style={{ cursor: 'pointer' }}
                    >
                      AMD
                    </Label>
                  </FlexItem>
                </Flex>
              </FlexItem>

              {/* Axis Selectors */}
              <FlexItem>
                <Flex direction={{ default: 'row' }} spaceItems={{ default: 'spaceItemsLg' }}>
                  <FlexItem flex={{ default: 'flex_1' }}>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                      X Axis:
                    </Text>
                    <ToggleGroup>
                      <ToggleGroupItem text="VRAM" isSelected={xAxis === 'vram'} onChange={() => setXAxis('vram')} />
                      {costMetricsReady && <ToggleGroupItem text="HW Cost" isSelected={xAxis === 'price'} onChange={() => setXAxis('price')} />}
                      <ToggleGroupItem text="Throughput Index" isSelected={xAxis === 'throughput-index'} onChange={() => setXAxis('throughput-index')} />
                      <ToggleGroupItem text="Mem BW" isSelected={xAxis === 'mem-bw'} onChange={() => setXAxis('mem-bw')} />
                    </ToggleGroup>
                  </FlexItem>

                  <FlexItem flex={{ default: 'flex_1' }}>
                    <Text component="p" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6a6e73', marginBottom: '8px', display: 'block', fontSize: '14px' }}>
                      Y Axis:
                    </Text>
                    <ToggleGroup>
                      <ToggleGroupItem text="VRAM" isSelected={yAxis === 'vram'} onChange={() => setYAxis('vram')} />
                      {costMetricsReady && <ToggleGroupItem text="HW Cost" isSelected={yAxis === 'price'} onChange={() => setYAxis('price')} />}
                      <ToggleGroupItem text="Throughput Index" isSelected={yAxis === 'throughput-index'} onChange={() => setYAxis('throughput-index')} />
                      <ToggleGroupItem text="Mem BW" isSelected={yAxis === 'mem-bw'} onChange={() => setYAxis('mem-bw')} />
                    </ToggleGroup>
                  </FlexItem>
                </Flex>
              </FlexItem>

              {/* Chart */}
              <FlexItem style={{ marginTop: '24px' }}>
                {!mounted || isLoading ? (
                  <div style={{ padding: '60px', textAlign: 'center', background: '#f5f5f5', borderRadius: '8px' }}>
                    <Text component="p" style={{ color: '#6a6e73' }}>Loading GPU catalog…</Text>
                  </div>
                ) : allData.length === 0 ? (
                  <div style={{ padding: '60px', textAlign: 'center', background: '#f5f5f5', borderRadius: '8px' }}>
                    <Text component="h3" style={{ color: '#6a6e73', marginBottom: '8px' }}>No data available</Text>
                    <Text component="p" style={{ color: '#6a6e73', fontSize: '14px' }}>Try selecting a different vendor filter</Text>
                  </div>
                ) : (
                  <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
                    <GpuBubbleChart
                      data={allData}
                      width={1100}
                      height={550}
                      xLabel={getAxisLabel(xAxis)}
                      yLabel={getAxisLabel(yAxis)}
                    />
                  </div>
                )}
              </FlexItem>

              {/* Legend */}
              <FlexItem>
                <Card isCompact>
                  <CardBody>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      💡 <strong>Top-right = high VRAM and throughput.</strong> These GPUs handle larger models and longer contexts.
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Bubble size</strong> represents {costingsEnabled && hasHardwareCosts
                        ? 'throughput index per $1k of hardware cost (a relative cost-efficiency metric)'
                        : 'BF16 TFLOPs'}.
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Throughput Index</strong> is a planning metric derived from memory bandwidth, VRAM, and architecture generation.
                      It enables relative GPU comparison — not exact model throughput.
                    </Text>
                    <Text component="p" style={{ display: 'block', marginBottom: '8px', color: '#3c3f42', fontSize: '13px', lineHeight: '1.6' }}>
                      <strong>Inference performance</strong> depends on model architecture (GQA vs MHA), sequence length, batching, and inference backend (vLLM, TensorRT-LLM, etc.).
                    </Text>
                    <Text component="p" style={{ display: 'block', color: '#6a6e73', fontSize: '13px', lineHeight: '1.6' }}>
                      {costingsEnabled
                        ? <><strong>Hardware cost</strong> axes and cost efficiency use live data from the costings API. Enable a cloud provider on the Sources page for cloud rates.</>
                        : <><strong>Hardware cost and cost efficiency</strong> axes appear when costings is enabled in Settings.</>}
                    </Text>
                  </CardBody>
                </Card>
              </FlexItem>
            </Flex>
          </CardBody>
        </Card>
      </PageSection>
    </>
  );
}
