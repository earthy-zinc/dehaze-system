/**
 * 一键生成原生 tabBar 图标（10 张 PNG）
 *
 * 来源：设计稿 dehaze-mobile/pages/tools-v2.html 底部导航的 5 个线性图标（heroicons 风格）
 * 输入：src/assets/tabbar/*.svg（灰色描边 #9CA3AF，即未选中态）
 * 输出：每个图标生成 2 张 PNG（81×81，小程序推荐尺寸）：
 *   - {name}.png          未选中态（#9CA3AF）
 *   - {name}-active.png   选中态（#3B82F6）
 *
 * 用法：node scripts/gen-tabbar-icons.mjs
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const tabbarDir = join(root, "src", "assets", "tabbar");
const ICON_SIZE = 81;
const INACTIVE = "#9CA3AF"; // 未选中：设计稿 --dehaze-neutral-400
const ACTIVE = "#3B82F6"; // 选中：设计稿 --dehaze-primary

const svgFiles = readdirSync(tabbarDir).filter((f) => f.endsWith(".svg"));

for (const file of svgFiles) {
  const name = file.replace(/\.svg$/, "");
  const svg = readFileSync(join(tabbarDir, file), "utf8");
  for (const [color, suffix] of [
    [INACTIVE, ""],
    [ACTIVE, "-active"],
  ]) {
    const coloredSvg = svg.replaceAll(INACTIVE, color);
    const png = await sharp(Buffer.from(coloredSvg))
      .resize(ICON_SIZE, ICON_SIZE)
      .png()
      .toBuffer();
    writeFileSync(join(tabbarDir, `${name}${suffix}.png`), png);
  }
  console.log(`✓ ${name} → ${name}.png + ${name}-active.png`);
}

console.log(`完成：${svgFiles.length} 个图标 × 2 态 = ${svgFiles.length * 2} 张 PNG`);
