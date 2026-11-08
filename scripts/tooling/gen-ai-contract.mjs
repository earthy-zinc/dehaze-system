#!/usr/bin/env node
/**
 * AI 对话契约生成器：config/schema/ai-conversation.schema.json → TypeScript / Python / Java / Go DTO。
 *
 * 契约以 JSON Schema 为唯一来源，四端只写实现、不手抄类型。改契约 = 改 Schema + 重跑本脚本。
 *
 * 用法:
 *   node scripts/tooling/gen-ai-contract.mjs
 *
 * 依赖:
 *   仅 Node 内置模块。若 PATH 中存在 gofmt，会对 Go 产物做一次就地格式化；
 *   缺少 gofmt 时产物仍合法，只是排版未对齐。
 *
 * 支持的 Schema 结构:
 *   $ref（仅 #/$defs/<Name>）、enum、anyOf: [X, null]、type: [X, "null"]、
 *   array/items、object/properties/required、additionalProperties: true|false。
 *   其余结构一律抛错，避免静默降级。
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCHEMA_REL = "config/schema/ai-conversation.schema.json";
const HEADER = "GENERATED FROM config/schema/ai-conversation.schema.json — DO NOT EDIT.";
const GENERATOR_NOTE =
  "由 scripts/tooling/gen-ai-contract.mjs 生成；改契约请改 Schema 后重新运行生成器。";

const OUT_TS = "clarity-sdk-js/src/api/ai-conversation/contract.gen.ts";
const OUT_PY = "clarity-python/app/models/schema/ai_contract_gen.py";
const OUT_JAVA = "clarity-java/src/main/java/com/pei/clarity/model/dto/ai/AiContract.java";
const OUT_GO = "clarity-go/internal/model/aicontract/contract.gen.go";

const REF_PREFIX = "#/$defs/";
const SCALAR_TYPES = new Set(["string", "integer", "number", "boolean"]);

// ==================== Schema → 中间表示 ====================

/** 取出 $ref 指向的 $defs 名称。 */
function refName(ref) {
  if (typeof ref !== "string" || !ref.startsWith(REF_PREFIX)) {
    throw new Error(`不支持的 $ref: ${JSON.stringify(ref)}（只接受 ${REF_PREFIX}<Name>）`);
  }
  return ref.slice(REF_PREFIX.length);
}

/** anyOf 里的 { type: "null" } 分支。 */
function isNullBranch(node) {
  return node !== null && typeof node === "object" && node.type === "null" && !node.$ref;
}

/**
 * 把一处 Schema 归一成 IR：
 *   { kind: "ref" | "enum" | "array" | "map" | "object" | "scalar", nullable, ... }
 * 引用节点用 `ref` 承载目标名，字段名始终是 `name`，两者不混用。
 */
function normalize(node, where) {
  if (node === null || typeof node !== "object" || Array.isArray(node)) {
    throw new Error(`${where}: Schema 节点必须是对象`);
  }
  if (node.$ref) {
    return { kind: "ref", ref: refName(node.$ref), nullable: false };
  }
  if (node.anyOf) {
    const branches = node.anyOf.filter((branch) => !isNullBranch(branch));
    if (branches.length !== 1) {
      throw new Error(`${where}: anyOf 只支持 [X, {type:"null"}] 形式`);
    }
    return { ...normalize(branches[0], where), nullable: true };
  }
  let type = node.type;
  let nullable = false;
  if (Array.isArray(type)) {
    nullable = type.includes("null");
    const rest = type.filter((item) => item !== "null");
    if (rest.length !== 1) {
      throw new Error(`${where}: type 数组只支持 [X, "null"] 形式`);
    }
    type = rest[0];
  }
  if (node.enum) {
    if (!SCALAR_TYPES.has(type)) {
      throw new Error(`${where}: enum 只支持标量类型，实际为 ${JSON.stringify(type)}`);
    }
    return { kind: "enum", values: node.enum, valueType: type, nullable };
  }
  if (type === "object") {
    const entries = Object.entries(node.properties ?? {});
    if (entries.length === 0) {
      return { kind: "map", additional: node.additionalProperties !== false, nullable };
    }
    const required = new Set(node.required ?? []);
    return {
      kind: "object",
      additional: node.additionalProperties === true,
      nullable,
      props: entries.map(([name, sub]) => ({
        ...normalize(sub, `${where}.${name}`),
        name,
        required: required.has(name),
      })),
    };
  }
  if (type === "array") {
    if (!node.items) {
      throw new Error(`${where}: array 缺少 items`);
    }
    return { kind: "array", items: normalize(node.items, `${where}[]`), nullable };
  }
  if (!SCALAR_TYPES.has(type)) {
    throw new Error(`${where}: 不支持的类型 ${JSON.stringify(type)}`);
  }
  return { kind: "scalar", type, nullable };
}

/** 收集 IR 中引用到的定义名（去重，保持出现顺序）。 */
function collectRefs(ir, bucket = []) {
  if (ir.kind === "ref") {
    if (!bucket.includes(ir.ref)) bucket.push(ir.ref);
  } else if (ir.kind === "array") {
    collectRefs(ir.items, bucket);
  } else if (ir.kind === "object") {
    for (const prop of ir.props) collectRefs(prop, bucket);
  }
  return bucket;
}

/** IR 中是否出现过某种节点（Java 判断需要 import 哪些集合类型）。 */
function hasKind(ir, kind) {
  if (ir.kind === kind) return true;
  if (ir.kind === "array") return hasKind(ir.items, kind);
  if (ir.kind === "object") return ir.props.some((prop) => hasKind(prop, kind));
  return false;
}

/** 读取 Schema，产出契约上下文（保持 $defs 声明顺序）。 */
function loadContract() {
  const schema = JSON.parse(readFileSync(resolve(REPO_ROOT, SCHEMA_REL), "utf8"));
  const rawDefs = schema.$defs;
  if (!rawDefs || typeof rawDefs !== "object") {
    throw new Error(`${SCHEMA_REL}: 缺少 $defs`);
  }
  const defs = Object.entries(rawDefs).map(([name, node]) => ({
    name,
    ir: normalize(node, name),
  }));
  const byName = new Map(defs.map((def) => [def.name, def]));
  for (const def of defs) {
    for (const ref of collectRefs(def.ir)) {
      if (!byName.has(ref)) {
        throw new Error(`${def.name}: 引用了未定义的类型 ${ref}`);
      }
    }
  }
  const enumValues = new Map(
    defs.filter((def) => def.ir.kind === "enum").map((def) => [def.name, def.ir.values]),
  );
  return { defs, byName, order: topoOrder(defs, byName), enumValues };
}

/** 依赖优先的稳定排序（Python 需要先定义后引用；其余端沿用同一顺序便于对读）。 */
function topoOrder(defs, byName) {
  const done = new Set();
  const order = [];
  const visit = (name) => {
    if (done.has(name)) return;
    done.add(name);
    for (const ref of collectRefs(byName.get(name).ir)) {
      if (ref !== name) visit(ref);
    }
    order.push(name);
  };
  for (const def of defs) visit(def.name);
  return order;
}

/** 枚举节点的取值（含 $ref 到枚举定义的情形），非枚举返回 null。 */
function enumValuesOf(ir, enumValues) {
  if (ir.kind === "enum") return ir.values;
  if (ir.kind === "ref" && enumValues.has(ir.ref)) return enumValues.get(ir.ref);
  return null;
}

// ==================== TypeScript ====================

const TS_SCALAR = { string: "string", integer: "number", number: "number", boolean: "boolean" };
const TS_WIDTH = 100;

function tsType(ir) {
  const base = tsBase(ir);
  return ir.nullable ? `${base} | null` : base;
}

function tsBase(ir) {
  switch (ir.kind) {
    case "ref":
      return ir.ref;
    case "enum":
      return ir.values.map((value) => JSON.stringify(value)).join(" | ");
    case "array": {
      const item = tsType(ir.items);
      const wrap =
        (ir.items.kind === "enum" && ir.items.values.length > 1) || ir.items.nullable === true;
      return `${wrap ? `(${item})` : item}[]`;
    }
    case "map":
      return "Record<string, unknown>";
    case "scalar":
      return TS_SCALAR[ir.type];
    default:
      throw new Error(`TS: 无法处理的节点 ${ir.kind}`);
  }
}

function emitTs(contract) {
  const blocks = [];
  for (const name of contract.order) {
    const ir = contract.byName.get(name).ir;
    if (ir.kind === "enum") {
      const values = ir.values.map((value) => JSON.stringify(value));
      const single = `export type ${name} = ${values.join(" | ")};`;
      if (single.length <= TS_WIDTH) {
        blocks.push(single);
        continue;
      }
      // 先尝试"换行后一行放完"（prettier 的 fill 风格），仍超宽才逐值竖排。
      const filled = `  ${values.join(" | ")};`;
      if (filled.length <= TS_WIDTH) {
        blocks.push(`export type ${name} =\n${filled}`);
        continue;
      }
      const lines = [`export type ${name} =`];
      values.forEach((value, index) => {
        lines.push(`  | ${value}${index === values.length - 1 ? ";" : ""}`);
      });
      blocks.push(lines.join("\n"));
      continue;
    }
    const lines = [`export interface ${name} {`];
    for (const prop of ir.props) {
      lines.push(`  ${prop.name}${prop.required ? "" : "?"}: ${tsType(prop)};`);
    }
    if (ir.additional) {
      lines.push("  [key: string]: unknown;");
    }
    lines.push("}");
    blocks.push(lines.join("\n"));
  }
  return ["/**", ` * ${HEADER}`, ` * ${GENERATOR_NOTE}`, " */", "", blocks.join("\n\n"), ""].join(
    "\n",
  );
}

// ==================== Python ====================

const PY_SCALAR = { string: "str", integer: "int", number: "float", boolean: "bool" };
const PY_WIDTH = 100;

function pyType(ir) {
  const base = pyBase(ir);
  return ir.nullable ? `${base} | None` : base;
}

function pyBase(ir) {
  switch (ir.kind) {
    case "ref":
      return ir.ref;
    case "enum":
      return `Literal[${ir.values.map((value) => pyLiteral(value)).join(", ")}]`;
    case "array":
      return `list[${pyType(ir.items)}]`;
    case "map":
      return "dict[str, Any]";
    case "scalar":
      return PY_SCALAR[ir.type];
    default:
      throw new Error(`Python: 无法处理的节点 ${ir.kind}`);
  }
}

/** Python 字面量：字符串用双引号，数字/布尔按 JSON 原样。 */
function pyLiteral(value) {
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

function emitPy(contract) {
  const blocks = [];
  for (const name of contract.order) {
    const ir = contract.byName.get(name).ir;
    if (ir.kind === "enum") {
      const values = ir.values.map((value) => pyLiteral(value));
      const single = `${name} = Literal[${values.join(", ")}]`;
      if (single.length <= PY_WIDTH) {
        blocks.push(single);
        continue;
      }
      blocks.push([`${name} = Literal[`, ...values.map((value) => `    ${value},`), "]"].join("\n"));
      continue;
    }
    const lines = [
      `class ${name}(BaseModel):`,
      `    model_config = ConfigDict(extra=${ir.additional ? '"allow"' : '"forbid"'})`,
      "",
    ];
    for (const prop of ir.props) {
      lines.push(`    ${prop.name}: ${pyType(prop)}${prop.required ? "" : " = None"}`);
    }
    blocks.push(lines.join("\n"));
  }
  return [
    `# ${HEADER}`,
    `# ${GENERATOR_NOTE}`,
    "# 字段名保持 camelCase，与 JSON 一致，不做 snake_case 转换。",
    "",
    "from __future__ import annotations",
    "",
    "from typing import Any, Literal",
    "",
    "from pydantic import BaseModel, ConfigDict",
    "",
    blocks.join("\n\n\n"),
    "",
  ].join("\n");
}

// ==================== Java ====================

const JAVA_SCALAR = {
  string: { primitive: "String", boxed: "String" },
  integer: { primitive: "long", boxed: "Long" },
  number: { primitive: "double", boxed: "Double" },
  boolean: { primitive: "boolean", boxed: "Boolean" },
};

/** Java 侧枚举一律 String（整数枚举保留整型），取值写在注释里；泛型参数必须是装箱类型。 */
function javaType(ir, isEnumDef, boxed = false) {
  switch (ir.kind) {
    case "ref":
      return isEnumDef ? "String" : ir.ref;
    case "enum":
      return ir.nullable || boxed ? JAVA_SCALAR[ir.valueType].boxed : JAVA_SCALAR[ir.valueType].primitive;
    case "array":
      return `List<${javaType(ir.items, isEnumDef, true)}>`;
    case "map":
      return "Map<String, Object>";
    case "scalar":
      return ir.nullable || boxed ? JAVA_SCALAR[ir.type].boxed : JAVA_SCALAR[ir.type].primitive;
    default:
      throw new Error(`Java: 无法处理的节点 ${ir.kind}`);
  }
}

function emitJava(contract) {
  const { order, byName, enumValues } = contract;
  const usesList = contract.defs.some((def) => hasKind(def.ir, "array"));
  const usesMap = contract.defs.some((def) => hasKind(def.ir, "map"));

  const lines = ["package com.pei.clarity.model.dto.ai;", ""];
  lines.push("import com.fasterxml.jackson.annotation.JsonInclude;");
  if (usesList || usesMap) {
    if (usesList) lines.push("import java.util.List;");
    if (usesMap) lines.push("import java.util.Map;");
  }
  lines.push("");
  lines.push(
    "/**",
    ` * ${HEADER}`,
    ` * ${GENERATOR_NOTE}`,
    " * 枚举字段一律为 String，取值见字段上方注释；时间字段一律为 String（ISO-8601）。",
    " * 每个 record 标注 @JsonInclude(ALWAYS)：嵌套结构的显式 null（durationMs/tool 等）不被",
    " * 全局 default-property-inclusion=non_null 省略，保证三端响应的键集合一致。",
    " */",
    "public final class AiContract {",
    "",
    "    private AiContract() {",
    "    }",
  );

  for (const name of order) {
    const ir = byName.get(name).ir;
    if (ir.kind !== "object") continue;
    const entries = ir.props.map((prop) => {
      const values = enumValuesOf(prop, enumValues);
      const comment = values ? `            // 取值: ${values.map(String).join(" | ")}` : null;
      const type = javaType(prop, prop.kind === "ref" && enumValues.has(prop.ref));
      const param = `            ${type} ${prop.name}`;
      return comment ? [comment, param] : [param];
    });
    const body = entries.map((entry, index) => {
      const last = index === entries.length - 1;
      return entry.map((line, i) => (i === entry.length - 1 && !last ? `${line},` : line)).join("\n");
    });
    lines.push("");
    if (ir.additional) {
      lines.push("    // 允许附加字段（Schema additionalProperties=true），实现侧自行透传");
    }
    // 嵌套 record 的 null 必须显式下发：全局 non_null 不会因外层 VO 的 ALWAYS 级联到 record 属性
    lines.push("    @JsonInclude(JsonInclude.Include.ALWAYS)");
    lines.push(`    public record ${name}(`, body.join("\n"), "    ) {", "    }");
  }

  lines.push("}", "");
  return lines.join("\n");
}

// ==================== Go ====================

const GO_SCALAR = { string: "string", integer: "int64", number: "float64", boolean: "bool" };
const GO_INITIALISMS = new Set(["ID", "URL", "API", "JSON", "HTTP", "LLM", "SSE", "MCP", "RAG"]);

/** camelCase → Go 导出名（Id → ID、Ids → IDs，保留常见缩写）。 */
function goFieldName(name) {
  return name
    .split(/(?=[A-Z])|_/)
    .filter(Boolean)
    .map((word) => {
      const upper = word.toUpperCase();
      if (GO_INITIALISMS.has(upper)) return upper;
      if (upper.endsWith("S") && GO_INITIALISMS.has(upper.slice(0, -1))) {
        return `${upper.slice(0, -1)}s`;
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join("");
}

/** snake_case 枚举值 → 导出常量后缀（awaiting_input → AwaitingInput）。 */
function goConstSuffix(value) {
  return String(value)
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join("");
}

function goType(ir, isEnumDef) {
  switch (ir.kind) {
    case "ref":
      return isEnumDef ? "string" : ir.ref;
    case "enum":
      return GO_SCALAR[ir.valueType];
    case "array":
      return `[]${goType(ir.items, isEnumDef)}`;
    case "map":
      return "map[string]any";
    case "scalar":
      return GO_SCALAR[ir.type];
    default:
      throw new Error(`Go: 无法处理的节点 ${ir.kind}`);
  }
}

/**
 * 可选字段：标量/结构用指针表达 null，切片/映射 nil 本身即 null。
 * 本项目契约里"可选"与"可空"等价，出站一律显式 null（不加 omitempty），便于三端逐字段对拍。
 */
function goFieldType(prop, isEnumDef) {
  const base = goType(prop, isEnumDef);
  if (prop.required) return base;
  if (prop.kind === "array" || prop.kind === "map") return base;
  return `*${base}`;
}

function emitGo(contract) {
  const { order, byName, enumValues } = contract;
  const constBlocks = [];
  const structs = [];
  for (const name of order) {
    const ir = byName.get(name).ir;
    if (ir.kind === "enum") {
      const lines = [`// ${name} 取值: ${ir.values.map(String).join(" | ")}`, "const ("];
      for (const value of ir.values) {
        lines.push(`\t${name}${goConstSuffix(value)} = ${JSON.stringify(value)}`);
      }
      lines.push(")");
      constBlocks.push(lines.join("\n"));
      continue;
    }
    const lines = [`type ${name} struct {`];
    for (const prop of ir.props) {
      const values = enumValuesOf(prop, enumValues);
      if (values) lines.push(`\t// 取值: ${values.map(String).join(" | ")}`);
      const isEnumDef = prop.kind === "ref" && enumValues.has(prop.ref);
      const tag = `json:"${prop.name}"`;
      lines.push(`\t${goFieldName(prop.name)} ${goFieldType(prop, isEnumDef)} \`${tag}\``);
    }
    if (ir.additional) {
      lines.push("\t// 允许附加字段（Schema additionalProperties=true），解码时按需自行保留");
    }
    lines.push("}");
    structs.push(lines.join("\n"));
  }
  return [
    `// ${HEADER}`,
    `// ${GENERATOR_NOTE}`,
    "// 枚举字段类型为 string，取值见 const 块与字段注释；时间字段为 string（ISO-8601）。",
    "",
    "package aicontract",
    "",
    constBlocks.join("\n\n"),
    "",
    structs.join("\n\n"),
    "",
  ].join("\n");
}

// ==================== 写盘 ====================

function write(relPath, content) {
  const absolute = resolve(REPO_ROOT, relPath);
  mkdirSync(dirname(absolute), { recursive: true });
  writeFileSync(absolute, content, "utf8");
  return relPath;
}

/** gofmt 存在时就地格式化；缺失时保留原样（仍可编译）。 */
function formatGo(relPath) {
  const result = spawnSync("gofmt", ["-w", resolve(REPO_ROOT, relPath)], { encoding: "utf8" });
  if (result.error) {
    process.stdout.write(`提示: 未找到 gofmt，${relPath} 保持未对齐排版\n`);
    return;
  }
  if (result.status !== 0) {
    throw new Error(`gofmt 失败: ${result.stderr || result.stdout}`);
  }
}

function main() {
  const contract = loadContract();
  const written = [
    write(OUT_TS, emitTs(contract)),
    write(OUT_PY, emitPy(contract)),
    write(OUT_JAVA, emitJava(contract)),
    write(OUT_GO, emitGo(contract)),
  ];
  formatGo(OUT_GO);
  process.stdout.write(`契约来源: ${SCHEMA_REL}（$defs ${contract.defs.length} 个）\n`);
  for (const path of written) {
    process.stdout.write(`已生成: ${path}\n`);
  }
}

main();
