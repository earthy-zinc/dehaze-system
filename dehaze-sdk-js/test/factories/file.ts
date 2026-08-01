import { FileInfo } from "@/api/file/model";

/**
 * 创建文件信息
 * @param overrides 覆盖默认值的字段
 */
export function createFileInfo(overrides: Partial<FileInfo> = {}): FileInfo {
  return {
    id: 1,
    name: "test_file_" + Date.now() + ".txt",
    type: "txt",
    size: "1.2KB",
    sizeBytes: 1234,
    url: "http://localhost:8989/api/v1/files/download/upload/20260101/test_file.txt",
    objectName: "upload/20260101/test_file.txt",
    storage: "minio",
    md5: "d41d8cd98f00b204e9800998ecf8427e",
    createTime: "2026-01-01T00:00:00",
    ...overrides,
  };
}
