/**
 * 测试资源保障：检测 test/resources/ 下的本地图片资源是否存在，缺失则调用
 * scripts/gen_test_resources.py 全量生成（脚本幂等，覆盖写入）。
 *
 * 缺失会导致 item-file.test.ts / model.test.ts / integration/* / recommendation
 * 等套件 beforeAll 用 fs.readFileSync 读取图片时 ENOENT，进而引发直接失败和
 * "所在套件钩子失败" 的级联跳过。此保障在 globalSetup（所有测试启动前）执行，
 * 从根因上消除该问题。
 *
 * 清单与 scripts/gen_test_resources.py 的 IMAGES/PNG_IMAGES（RESOURCES_DIR 部分）保持同步。
 */
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

// 与 scripts/gen_test_resources.py 的 IMAGES/PNG_IMAGES 对齐（仅本地 resources 部分，
// 不含写入项目根 datasets/ 的 DATASET_IMAGES——后者经 nginx 容器挂载按 URL 访问）
const RESOURCE_REL_PATHS = [
  "test/clean/41_outdoor_GT.jpg",
  "test/clean/42_outdoor_GT.jpg",
  "test/clean/43_outdoor_GT.jpg",
  "test/clean/44_outdoor_GT.jpg",
  "test/clean/45_outdoor_GT.jpg",
  "test/hazy/41_outdoor_hazy.jpg",
  "test/hazy/42_outdoor_hazy.jpg",
  "test/hazy/43_outdoor_hazy.jpg",
  "test2/clean/0025.jpg",
  "test2/hazy/0025_0.8_0.04.jpg",
  "test2/hazy/0025_0.8_0.08.jpg",
  "test2/hazy/0025_0.9_0.12.jpg",
  "test/model/hazy.jpg",
  "test/model/clear.jpg",
  "test3/cqupt.png",
];

/** 优先使用含 Pillow/numpy 的 venv python，兜底用系统 python3 */
function resolvePython(projectRoot: string): string {
  const venvPython = path.join(projectRoot, "dehaze-python/.venv/bin/python");
  return fs.existsSync(venvPython) ? venvPython : "python3";
}

export function ensureTestResources(): void {
  const resourcesDir = path.resolve(__dirname, "../resources");
  const missing = RESOURCE_REL_PATHS.filter((rel) => !fs.existsSync(path.join(resourcesDir, rel)));
  if (missing.length === 0) return;

  const projectRoot = path.resolve(__dirname, "../../..");
  const script = path.join(projectRoot, "scripts/gen_test_resources.py");
  try {
    const stdout = execFileSync(resolvePython(projectRoot), [script], { encoding: "utf8" });
    console.log(`[ensure-test-resources] 已生成缺失测试资源 (${missing.length} 个):\n${stdout}`);
  } catch (e) {
    console.error(
      `[ensure-test-resources] 资源生成失败，请手动执行 python scripts/gen_test_resources.py:\n${
        (e as Error).message
      }`
    );
  }
}
