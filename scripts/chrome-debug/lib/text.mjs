// 模板渲染：把 {{key}} 替换成 values[key]，取值缺失时替换为空串。
// 替换值走函数形式，避免值里的 $& / $1 被当成替换模式。
export function render(template, values) {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
    const value = values[key];
    if (value === undefined || value === null) {
      return '';
    }
    return value;
  });
}
