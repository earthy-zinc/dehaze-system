/**
 * 文件哈希计算工具
 *
 * 浏览器原生不支持 MD5 算法，且项目未引入 SparkMD5 依赖。
 * 此处使用 Web Crypto API 的 crypto.subtle.digest 计算 SHA-256 哈希作为文件唯一标识，
 * 用于上传前的秒传校验。
 */

/**
 * 计算文件哈希值（基于文件内容，使用 SHA-256 算法）
 *
 * @param file 文件对象
 * @returns 十六进制哈希字符串
 */
export async function calculateFileMd5(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
