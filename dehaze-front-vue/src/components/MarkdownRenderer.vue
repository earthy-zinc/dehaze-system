<template>
  <div
    ref="containerRef"
    class="markdown-body"
    @click="handleClick"
    v-html="html"
  />
</template>

<script lang="ts" setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import MarkdownIt from "markdown-it";
import katex from "katex";
import "katex/dist/katex.min.css";
import texmath from "markdown-it-texmath";
import mermaid from "mermaid";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import csharp from "highlight.js/lib/languages/csharp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import dockerfile from "highlight.js/lib/languages/dockerfile";
import go from "highlight.js/lib/languages/go";
import graphql from "highlight.js/lib/languages/graphql";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import kotlin from "highlight.js/lib/languages/kotlin";
import lua from "highlight.js/lib/languages/lua";
import markdown from "highlight.js/lib/languages/markdown";
import php from "highlight.js/lib/languages/php";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import shell from "highlight.js/lib/languages/shell";
import sql from "highlight.js/lib/languages/sql";
import swift from "highlight.js/lib/languages/swift";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

defineOptions({ name: "MarkdownRenderer" });

const props = defineProps<{
  content: string;
}>();

mermaid.initialize({ startOnLoad: false });

const FOLD_LINE_THRESHOLD = 20;
const RENDER_INTERVAL_MS = 80;

// 按需注册常用语言，避免引入全量 highlight.js 体积
[
  bash,
  c,
  cpp,
  csharp,
  css,
  diff,
  dockerfile,
  go,
  graphql,
  java,
  javascript,
  json,
  kotlin,
  lua,
  markdown,
  php,
  plaintext,
  python,
  rust,
  shell,
  sql,
  swift,
  typescript,
  xml,
  yaml,
].forEach((lang) => hljs.registerLanguage(lang.name, lang));

function isSafeUrl(url: string): boolean {
  const trimmed = url.trim().toLowerCase();
  return !/^(javascript|vbscript|data):/i.test(trimmed);
}

function createMarkdownRenderer(isBlockExpanded: (index: number) => boolean) {
  const md = new MarkdownIt({
    html: false, // LLM 输出不可信，原生 HTML 一律转义
    linkify: true,
    breaks: true,
  });
  // 数学公式：$...$ 行内、$$...$$ 块级（KaTeX 同步渲染，出错不抛异常）
  md.use(texmath, {
    engine: katex,
    delimiters: "dollars",
    katexOptions: { throwOnError: false, output: "html" },
  });

  // GFM 任务列表：把列表项开头的 [ ]/[x] 文本替换为禁用态 checkbox（html:false 下用 html_inline 注入）
  md.core.ruler.after("inline", "task_list", (state) => {
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type !== "list_item_open") continue;
      const inline = tokens[i + 2];
      if (
        tokens[i + 1]?.type !== "paragraph_open" ||
        inline?.type !== "inline"
      ) {
        continue;
      }
      const first = inline.children?.[0];
      if (!first || first.type !== "text") continue;
      const match = /^\[([ xX])\] +/.exec(first.content);
      if (!match) continue;
      first.content = first.content.slice(match[0].length);
      const checkbox = new state.Token("html_inline", "", 0);
      checkbox.content = `<input type="checkbox" class="md-task-checkbox" disabled${
        match[1] === " " ? "" : " checked"
      }>`;
      inline.children?.unshift(checkbox);
      tokens[i].attrJoin("class", "task-list-item");
    }
  });

  let codeBlockSeq = 0;

  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const code = token.content;
    const lang = (token.info || "").trim().split(/\s+/)[0].toLowerCase();
    if (lang === "mermaid") {
      // Mermaid 图：先以源码渲染（可折叠），DOM 更新后由 mermaid.run 异步渲染为 SVG
      return (
        `<div class="mermaid" data-code="${encodeURIComponent(code)}">` +
        `<pre><code>${md.utils.escapeHtml(code)}</code></pre>` +
        "</div>\n"
      );
    }
    const blockIndex = codeBlockSeq++;
    const lineCount = code ? code.replace(/\n$/, "").split("\n").length : 1;
    const foldable = lineCount > FOLD_LINE_THRESHOLD;
    const folded = foldable && !isBlockExpanded(blockIndex);

    const highlighted =
      lang && hljs.getLanguage(lang)
        ? hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
        : md.utils.escapeHtml(code);

    return (
      `<div class="md-code-block${folded ? " md-folded" : ""}" data-index="${blockIndex}">` +
      '<div class="md-code-header">' +
      `<span class="md-code-lang">${md.utils.escapeHtml(lang || "text")}</span>` +
      '<button class="md-copy-btn" type="button">复制</button>' +
      "</div>" +
      `<pre><code class="hljs language-${md.utils.escapeHtml(lang)}">${highlighted}</code></pre>` +
      (foldable
        ? `<button class="md-fold-toggle" type="button">${
            folded ? `展开全部（${lineCount} 行）` : "收起"
          }</button>`
        : "") +
      "</div>\n"
    );
  };

  md.renderer.rules.link_open = (tokens, idx, options) => {
    const href = String(tokens[idx].attrGet("href") ?? "");
    if (!isSafeUrl(href)) {
      tokens[idx].attrSet("href", "#");
    }
    tokens[idx].attrSet("target", "_blank");
    tokens[idx].attrSet("rel", "noopener noreferrer");
    return md.renderer.renderToken(tokens, idx, options);
  };

  md.renderer.rules.image = (tokens, idx, options) => {
    tokens[idx].attrSet("loading", "lazy");
    tokens[idx].attrSet("decoding", "async");
    return md.renderer.renderToken(tokens, idx, options);
  };

  md.renderer.rules.table_open = () => '<div class="md-table-wrap"><table>';
  md.renderer.rules.table_close = () => "</table></div>";

  return {
    render(src: string): string {
      codeBlockSeq = 0;
      return md.render(src);
    },
  };
}

const containerRef = ref<HTMLElement>();
const expandedBlocks = ref(new Set<number>());
const renderer = createMarkdownRenderer((index) =>
  expandedBlocks.value.has(index)
);

// 流式场景：80ms 尾随节流合并 props 高频更新，避免每 token 全量重解析
const pendingContent = ref(props.content);
const html = computed(() => renderer.render(pendingContent.value));

let latestContent = props.content;
let renderTimer: ReturnType<typeof setTimeout> | null = null;
let lastRenderAt = 0;

function escapeHtmlText(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Mermaid 异步渲染：仅渲染未完成的图节点，失败恢复源码等待下次（流式中间态图不完整）。 */
async function renderMermaid() {
  const container = containerRef.value;
  if (!container) return;
  const nodes = Array.from(
    container.querySelectorAll<HTMLElement>(".mermaid:not([data-rendered])")
  );
  for (const node of nodes) {
    try {
      await mermaid.run({ nodes: [node] });
      node.setAttribute("data-rendered", "1");
    } catch {
      const code = node.getAttribute("data-code");
      if (code) {
        node.innerHTML = `<pre><code>${escapeHtmlText(decodeURIComponent(code))}</code></pre>`;
      }
    }
  }
}

function flushRender() {
  renderTimer = null;
  lastRenderAt = Date.now();
  pendingContent.value = latestContent;
  void nextTick(renderMermaid);
}

watch(
  () => props.content,
  (val) => {
    latestContent = val;
    if (renderTimer !== null) return;
    const wait = Math.max(0, RENDER_INTERVAL_MS - (Date.now() - lastRenderAt));
    renderTimer = setTimeout(flushRender, wait);
  }
);

onBeforeUnmount(() => {
  if (renderTimer !== null) {
    clearTimeout(renderTimer);
    renderTimer = null;
  }
});

// 代码块复制/折叠通过事件委托处理（v-html 内部无法绑定 Vue 事件）
function toggleFold(index: number) {
  if (expandedBlocks.value.has(index)) {
    expandedBlocks.value.delete(index);
  } else {
    expandedBlocks.value.add(index);
  }
}

// navigator.clipboard 仅在安全上下文可用，非安全上下文降级 execCommand
function legacyCopy(text: string): boolean {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);
  return copied;
}

async function handleClick(event: MouseEvent) {
  const target = event.target as HTMLElement;
  const copyBtn = target.closest(".md-copy-btn");
  if (copyBtn) {
    const code =
      copyBtn.closest(".md-code-block")?.querySelector("pre code")
        ?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(code);
      ElMessage.success("代码已复制");
    } catch {
      if (legacyCopy(code)) {
        ElMessage.success("代码已复制");
      } else {
        ElMessage.error("复制失败，请手动选择复制");
      }
    }
    return;
  }
  const foldBtn = target.closest(".md-fold-toggle");
  if (foldBtn) {
    const index = Number(
      foldBtn.closest(".md-code-block")?.getAttribute("data-index")
    );
    if (!Number.isNaN(index)) {
      toggleFold(index);
    }
  }
}
</script>

<style lang="scss" scoped>
.markdown-body {
  font-size: 14px;
  line-height: 1.75;
  color: var(--el-text-color-primary);
  overflow-wrap: break-word;

  :deep(h1),
  :deep(h2),
  :deep(h3),
  :deep(h4),
  :deep(h5),
  :deep(h6) {
    margin: 1em 0 0.5em;
    font-weight: 600;
    line-height: 1.4;
  }

  :deep(h1) {
    font-size: 1.5em;
  }

  :deep(h2) {
    font-size: 1.3em;
  }

  :deep(h3) {
    font-size: 1.15em;
  }

  :deep(h4),
  :deep(h5),
  :deep(h6) {
    font-size: 1em;
  }

  :deep(> :first-child) {
    margin-top: 0;
  }

  :deep(> :last-child) {
    margin-bottom: 0;
  }

  :deep(p) {
    margin: 0.5em 0;
  }

  :deep(ul),
  :deep(ol) {
    padding-left: 1.5em;
    margin: 0.5em 0;
  }

  :deep(li) {
    margin: 0.25em 0;
  }

  :deep(li.task-list-item) {
    list-style: none;

    .md-task-checkbox {
      margin-right: 6px;
      pointer-events: none;
    }
  }

  :deep(blockquote) {
    padding: 4px 12px;
    margin: 0.5em 0;
    color: var(--el-text-color-secondary);
    background: var(--el-fill-color-lighter);
    border-left: 3px solid var(--el-border-color);
    border-radius: 0 4px 4px 0;
  }

  :deep(hr) {
    margin: 1em 0;
    border: none;
    border-top: 1px solid var(--el-border-color-lighter);
  }

  :deep(a) {
    color: var(--el-color-primary);
    text-decoration: none;

    &:hover {
      text-decoration: underline;
    }
  }

  :deep(img) {
    max-width: 100%;
    border-radius: 4px;
  }

  :deep(code) {
    padding: 2px 6px;
    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.9em;
    background: var(--el-fill-color);
    border-radius: 4px;
  }

  :deep(.md-table-wrap) {
    margin: 0.5em 0;
    overflow-x: auto;

    table {
      border-collapse: collapse;

      th,
      td {
        padding: 6px 12px;
        border: 1px solid var(--el-border-color-lighter);
      }

      th {
        font-weight: 600;
        background: var(--el-fill-color-light);
      }
    }
  }

  :deep(.mermaid) {
    margin: 0.5em 0;
    overflow-x: auto;
    text-align: center;

    svg {
      max-width: 100%;
      height: auto;
    }
  }

  :deep(.md-code-block) {
    margin: 0.5em 0;
    overflow: hidden;
    border: 1px solid var(--el-border-color-lighter);
    border-radius: 6px;

    pre {
      padding: 12px 16px;
      margin: 0;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.6;
      background: #f6f8fa;

      code {
        padding: 0;
        background: transparent;
      }
    }

    &.md-folded pre {
      max-height: 300px;
      overflow: hidden;
    }

    .hljs-comment,
    .hljs-quote {
      font-style: italic;
      color: #6a737d;
    }

    .hljs-keyword,
    .hljs-selector-tag,
    .hljs-literal,
    .hljs-doctag,
    .hljs-name {
      color: #d73a49;
    }

    .hljs-string,
    .hljs-regexp,
    .hljs-addition,
    .hljs-meta {
      color: #032f62;
    }

    .hljs-number,
    .hljs-built_in,
    .hljs-builtin-name,
    .hljs-variable,
    .hljs-template-variable,
    .hljs-attr,
    .hljs-attribute,
    .hljs-selector-attr,
    .hljs-selector-class,
    .hljs-selector-id,
    .hljs-link {
      color: #005cc5;
    }

    .hljs-title,
    .hljs-title.class_,
    .hljs-title.function_,
    .hljs-section,
    .hljs-type {
      color: #6f42c1;
    }

    .hljs-tag {
      color: #22863a;
    }

    .hljs-symbol,
    .hljs-bullet {
      color: #e36209;
    }

    .hljs-deletion {
      color: #b31d28;
    }

    .hljs-emphasis {
      font-style: italic;
    }

    .hljs-strong {
      font-weight: 600;
    }
  }

  :deep(.md-code-header) {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    font-size: 12px;
    background: var(--el-fill-color-light);
    border-bottom: 1px solid var(--el-border-color-lighter);
  }

  :deep(.md-code-lang) {
    color: var(--el-text-color-secondary);
  }

  :deep(.md-copy-btn) {
    padding: 2px 8px;
    color: var(--el-color-primary);
    cursor: pointer;
    background: transparent;
    border: none;
    border-radius: 4px;

    &:hover {
      background: var(--el-fill-color);
    }
  }

  :deep(.md-fold-toggle) {
    display: block;
    width: 100%;
    padding: 4px 0;
    font-size: 12px;
    color: var(--el-color-primary);
    text-align: center;
    cursor: pointer;
    background: transparent;
    border: none;
    border-top: 1px dashed var(--el-border-color-lighter);

    &:hover {
      background: var(--el-fill-color-light);
    }
  }
}
</style>
