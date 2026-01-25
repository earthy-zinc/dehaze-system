import { FileUploadCheckQuery } from "@/api/file/model";

/**
 * 创建文件上传检查查询参数
 * @param overrides 覆盖默认值的字段
 */
export function createFileUploadCheckQuery(
  overrides: Partial<FileUploadCheckQuery> = {}
): FileUploadCheckQuery {
  return {
    md5: "test_md5_" + Date.now(),
    ...overrides,
  };
}
