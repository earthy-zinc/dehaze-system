import type {
  ExportModule,
  ExportRequest,
  ImportModule,
  ImportRequest,
} from "@/api/import-export/model";

export function createExportRequest(overrides: Partial<ExportRequest> = {}): ExportRequest {
  return {
    format: "excel",
    ...overrides,
  };
}

export function createImportRequest(overrides: Partial<ImportRequest> = {}): ImportRequest {
  return {
    mode: "all",
    ...overrides,
  };
}

export function createCsvExportRequest(overrides: Partial<ExportRequest> = {}): ExportRequest {
  return createExportRequest({ format: "csv", ...overrides });
}

export function createAsyncExportRequest(overrides: Partial<ExportRequest> = {}): ExportRequest {
  return createExportRequest({ async: true, ...overrides });
}

export function createPartialImportRequest(overrides: Partial<ImportRequest> = {}): ImportRequest {
  return createImportRequest({ mode: "partial", ...overrides });
}

const MINIMAL_CSV_CONTENT =
  "username,nickname,email,mobile,gender,status,deptId\n" +
  "test_sdk_user,SDK测试用户,test_sdk@example.com,13800000001,1,1,1\n";

export function createCsvFile(name = "import_test.csv", content = MINIMAL_CSV_CONTENT): File {
  return new File([content], name, { type: "text/csv" });
}

export function createInvalidFile(name = "import_test.txt"): File {
  return new File([new Uint8Array([0x00, 0x01, 0x02])], name, {
    type: "text/plain",
  });
}

export function createOversizedFile(name = "import_test.xlsx"): File {
  const size = 21 * 1024 * 1024;
  const blob = new Blob([new Uint8Array(size)]);
  return new File([blob], name, {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}

export function isBlobLike(value: unknown): boolean {
  if (value instanceof Blob) return true;
  const obj = value as { size?: number; type?: string } | null;
  return (
    !!obj && typeof obj === "object" && typeof obj.size === "number" && typeof obj.type === "string"
  );
}

export function isTaskResultLike(value: unknown): boolean {
  const obj = value as { taskId?: unknown; status?: unknown } | null;
  return !!obj && typeof obj === "object" && typeof obj.taskId === "string" && "status" in obj;
}
