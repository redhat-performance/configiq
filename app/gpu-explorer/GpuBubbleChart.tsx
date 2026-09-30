"use client";

import * as React from 'react';

interface BubbleData {
  x: number;
  y: number;
  size: number;
  name: string;
  color: string;
  fullName?: string;
  vram?: number;
  hwCost?: number;
  memBW?: number;
  tokensPerDollar?: number;
  architecture?: string;
  tflops?: number;
}

interface Props {
  data: BubbleData[];
  width: number;
  height: number;
  xLabel: string;
  yLabel: string;
}

export function GpuBubbleChart({ data, width, height, xLabel, yLabel }: Props) {
  const [hoveredIndex, setHoveredIndex] = React.useState<number | null>(null);
  const [mousePos, setMousePos] = React.useState({ x: 0, y: 0 });
  const svgRef = React.useRef<SVGSVGElement>(null);

  const padding = { top: 40, right: 40, bottom: 60, left: 70 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      setMousePos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  // Calculate scales
  const xValues = data.map(d => d.x);
  const yValues = data.map(d => d.y);
  const sizeValues = data.map(d => d.size);

  const xMin = 0;
  const rawXMax = Math.max(0, ...xValues);
  const xMax = Number.isFinite(rawXMax) && rawXMax > 0 ? rawXMax * 1.1 : 1;
  const yMin = 0;
  const rawYMax = Math.max(0, ...yValues);
  const yMax = Number.isFinite(rawYMax) && rawYMax > 0 ? rawYMax * 1.1 : 1;
  const sizeMin = sizeValues.length === 0 ? 0 : Math.min(...sizeValues);
  const sizeMax = sizeValues.length === 0 ? 0 : Math.max(...sizeValues);
  const rawSizeRange = sizeMax - sizeMin;
  const sizeRange = Number.isFinite(rawSizeRange) && rawSizeRange > 0 ? rawSizeRange : 1;

  // Scale functions
  const scaleX = (val: number) => (val / xMax) * chartWidth;
  const scaleY = (val: number) => chartHeight - (val / yMax) * chartHeight;
  const scaleSize = (val: number) => {
    const normalized = (val - sizeMin) / sizeRange;
    return 5 + normalized * 25;
  };

  // Generate tick values
  const xTicks = Array.from({ length: 6 }, (_, i) => (xMax / 5) * i);
  const yTicks = Array.from({ length: 6 }, (_, i) => (yMax / 5) * i);

  // Collision detection for labels
  const checkLabelCollision = (i: number) => {
    const point = data[i];
    const cx = padding.left + scaleX(point.x);
    const cy = padding.top + scaleY(point.y);
    const r = scaleSize(point.size);
    const labelY = cy - r - 5;

    // Check against all other bubbles
    for (let j = 0; j < data.length; j++) {
      if (i === j) continue;
      const other = data[j];
      const otherCx = padding.left + scaleX(other.x);
      const otherCy = padding.top + scaleY(other.y);
      const otherR = scaleSize(other.size);
      const otherLabelY = otherCy - otherR - 5;

      // Check if labels would overlap (approximate check)
      const dx = Math.abs(cx - otherCx);
      const dy = Math.abs(labelY - otherLabelY);
      const nameWidth = point.name.length * 6.5; // Approximate character width at 11px
      const otherNameWidth = other.name.length * 6.5;

      if (dx < (nameWidth + otherNameWidth) / 2 && dy < 14) {
        return true; // Collision detected
      }
    }
    return false;
  };

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ background: '#fff', borderRadius: '8px' }}
      onMouseMove={handleMouseMove}
    >
      {/* Grid lines */}
      {xTicks.map((tick, i) => (
        <line
          key={`x-grid-${i}`}
          x1={padding.left + scaleX(tick)}
          y1={padding.top}
          x2={padding.left + scaleX(tick)}
          y2={padding.top + chartHeight}
          stroke="#e0e0e0"
          strokeWidth={1}
        />
      ))}
      {yTicks.map((tick, i) => (
        <line
          key={`y-grid-${i}`}
          x1={padding.left}
          y1={padding.top + scaleY(tick)}
          x2={padding.left + chartWidth}
          y2={padding.top + scaleY(tick)}
          stroke="#e0e0e0"
          strokeWidth={1}
        />
      ))}

      {/* X Axis */}
      <line
        x1={padding.left}
        y1={padding.top + chartHeight}
        x2={padding.left + chartWidth}
        y2={padding.top + chartHeight}
        stroke="#151515"
        strokeWidth={2}
      />
      {xTicks.map((tick, i) => (
        <g key={`x-tick-${i}`}>
          <line
            x1={padding.left + scaleX(tick)}
            y1={padding.top + chartHeight}
            x2={padding.left + scaleX(tick)}
            y2={padding.top + chartHeight + 5}
            stroke="#151515"
            strokeWidth={1}
          />
          <text
            x={padding.left + scaleX(tick)}
            y={padding.top + chartHeight + 20}
            textAnchor="middle"
            fontSize={12}
            fill="#3c3f42"
            fontFamily="var(--font-sans, sans-serif)"
          >
            {Math.round(tick)}
          </text>
        </g>
      ))}
      <text
        x={padding.left + chartWidth / 2}
        y={height - 10}
        textAnchor="middle"
        fontSize={13}
        fontWeight={500}
        fill="#151515"
        fontFamily="var(--font-sans, sans-serif)"
      >
        {xLabel}
      </text>

      {/* Y Axis */}
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={padding.top + chartHeight}
        stroke="#151515"
        strokeWidth={2}
      />
      {yTicks.map((tick, i) => (
        <g key={`y-tick-${i}`}>
          <line
            x1={padding.left - 5}
            y1={padding.top + scaleY(tick)}
            x2={padding.left}
            y2={padding.top + scaleY(tick)}
            stroke="#151515"
            strokeWidth={1}
          />
          <text
            x={padding.left - 10}
            y={padding.top + scaleY(tick)}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={12}
            fill="#3c3f42"
            fontFamily="var(--font-sans, sans-serif)"
          >
            {Math.round(tick).toLocaleString()}
          </text>
        </g>
      ))}
      <text
        x={15}
        y={padding.top + chartHeight / 2}
        textAnchor="middle"
        fontSize={13}
        fontWeight={500}
        fill="#151515"
        fontFamily="var(--font-sans, sans-serif)"
        transform={`rotate(-90, 15, ${padding.top + chartHeight / 2})`}
      >
        {yLabel}
      </text>

      {/* Bubbles */}
      {data.map((point, i) => {
        const cx = padding.left + scaleX(point.x);
        const cy = padding.top + scaleY(point.y);
        const r = scaleSize(point.size);
        const hasCollision = checkLabelCollision(i);
        const isHovered = hoveredIndex === i;

        return (
          <g key={i}>
            <circle
              cx={cx}
              cy={cy}
              r={r}
              fill={point.color}
              stroke="#fff"
              strokeWidth={2}
              opacity={isHovered ? 0.95 : 0.75}
              style={{ cursor: 'pointer', transition: 'opacity 150ms' }}
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
            {/* Only show label if no collision or if hovered */}
            {(!hasCollision || isHovered) && (
              <text
                x={cx}
                y={cy - r - 5}
                textAnchor="middle"
                fontSize={11}
                fontWeight={600}
                fill="#151515"
                style={{ pointerEvents: 'none' }}
              >
                {point.name}
              </text>
            )}
          </g>
        );
      })}

      {/* Tooltip - follows mouse cursor */}
      {hoveredIndex !== null && (() => {
        const tooltipWidth = 260;
        const tooltipHeight = 180;
        const tooltipX = mousePos.x + 15 > width - tooltipWidth ? mousePos.x - tooltipWidth - 15 : mousePos.x + 15;
        const tooltipY = mousePos.y - 90 < 0 ? mousePos.y + 20 : mousePos.y - 90;

        return (
          <g style={{ pointerEvents: 'none' }}>
            <foreignObject
              x={tooltipX}
              y={tooltipY}
              width={tooltipWidth}
              height={tooltipHeight}
            >
              <div style={{
                background: 'rgba(255, 255, 255, 0.98)',
                border: '2px solid #151515',
                borderRadius: '8px',
                padding: '14px 16px',
                fontSize: '13px',
                lineHeight: '1.7',
                boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                fontFamily: 'var(--pf-t--global--font--family--body, "Red Hat Text", sans-serif)',
                width: '100%',
                height: '100%',
                boxSizing: 'border-box'
              }}>
                <div style={{ fontWeight: 700, fontSize: '15px', marginBottom: '10px', color: '#151515', borderBottom: '1px solid #d2d2d2', paddingBottom: '8px' }}>
                  {data[hoveredIndex].fullName || data[hoveredIndex].name}
                </div>
                <div style={{ color: '#3c3f42', fontSize: '12px' }}>
                  <div style={{ marginBottom: '3px' }}><strong>VRAM:</strong> {data[hoveredIndex].vram} GB</div>
                  <div style={{ marginBottom: '3px' }}><strong>Memory BW:</strong> {data[hoveredIndex].memBW?.toFixed(1)} GB/s</div>
                  {data[hoveredIndex].hwCost != null && (
                    <div style={{ marginBottom: '3px' }}><strong>Hardware cost:</strong> ${data[hoveredIndex].hwCost?.toLocaleString()}</div>
                  )}
                  <div style={{ marginBottom: '3px' }}><strong>Architecture:</strong> {data[hoveredIndex].architecture}</div>
                  <div style={{ marginBottom: data[hoveredIndex].tokensPerDollar ? '3px' : 0 }}><strong>TFLOPS (BF16):</strong> {data[hoveredIndex].tflops}</div>
                  {data[hoveredIndex].tokensPerDollar ? (
                    <div><strong>Throughput / $1k:</strong> {Math.round(data[hoveredIndex].tokensPerDollar!).toLocaleString()}</div>
                  ) : null}
                </div>
              </div>
            </foreignObject>
          </g>
        );
      })()}
    </svg>
  );
}
