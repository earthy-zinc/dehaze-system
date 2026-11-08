/**
 * 把 CSS 的 linear-gradient(...) 解析成 LinearGradient 需要的参数。
 *
 * 支持项目里出现的两种写法：
 *   linear-gradient(180deg, #eff6ff 0%, #fff 100%)
 *   linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%)
 * 角度省略时按 CSS 默认（to bottom）处理。
 */
export type GradientStop = { color: string; location?: number };

export type ParsedGradient = {
  colors: string[];
  locations: number[];
  start: { x: number; y: number };
  end: { x: number; y: number };
};

const GRADIENT_RE = /^linear-gradient\((.*)\)$/i;
// 颜色可以是 #rgb / #rrggbb / rgb() / rgba()，位置是可选的百分比
const STOP_RE = /(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\))\s*([\d.]+%)?/g;

function angleToPoints(deg: number) {
  // CSS 角度：0deg 向上、90deg 向右、180deg 向下
  const rad = ((deg - 90) * Math.PI) / 180;
  const dx = Math.cos(rad) / 2;
  const dy = Math.sin(rad) / 2;
  return {
    start: { x: 0.5 - dx, y: 0.5 - dy },
    end: { x: 0.5 + dx, y: 0.5 + dy },
  };
}

export function parseGradient(value?: string | null): ParsedGradient | null {
  if (typeof value !== "string") return null;
  const matched = GRADIENT_RE.exec(value.trim());
  if (!matched) return null;

  const body = matched[1];
  // 第一个逗号前可能是角度或方向词
  const firstComma = body.indexOf(",");
  if (firstComma < 0) return null;

  const head = body.slice(0, firstComma).trim();
  const rest = body.slice(firstComma + 1);

  let points = angleToPoints(180);
  const degMatch = /^(-?[\d.]+)deg$/.exec(head);
  if (degMatch) {
    points = angleToPoints(parseFloat(degMatch[1]));
  } else if (/^to\s+right$/i.test(head)) {
    points = { start: { x: 0, y: 0.5 }, end: { x: 1, y: 0.5 } };
  } else if (/^to\s+left$/i.test(head)) {
    points = { start: { x: 1, y: 0.5 }, end: { x: 0, y: 0.5 } };
  } else if (/^to\s+top$/i.test(head)) {
    points = { start: { x: 0.5, y: 1 }, end: { x: 0.5, y: 0 } };
  }

  const stops: GradientStop[] = [];
  let match: RegExpExecArray | null;
  STOP_RE.lastIndex = 0;
  while ((match = STOP_RE.exec(rest))) {
    const stop: GradientStop = { color: match[1] };
    if (match[2]) stop.location = parseFloat(match[2]) / 100;
    stops.push(stop);
  }
  if (stops.length < 2) return null;

  // LinearGradient 要求 locations 与 colors 等长且单调递增，缺省位置按均分补齐
  const locations = stops.map((stop, index) =>
    stop.location === undefined ? index / (stops.length - 1) : stop.location
  );

  return {
    colors: stops.map((stop) => stop.color),
    locations,
    start: points.start,
    end: points.end,
  };
}
