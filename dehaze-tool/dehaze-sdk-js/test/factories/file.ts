import { FileInfo } from "@/api/file/model";

/**
 * 创建文件信息
 * @param overrides 覆盖默认值的字段
 */
export function createFileInfo(
  overrides: Partial<FileInfo> = {}
): FileInfo {
  return {
    id: 1,
    name: "test_file_" + Date.now() + ".txt",
    path: "/test/test_file.txt",
    url: "http://localhost:8989/test/test_file.txt",
    ...overrides,
  };
}
