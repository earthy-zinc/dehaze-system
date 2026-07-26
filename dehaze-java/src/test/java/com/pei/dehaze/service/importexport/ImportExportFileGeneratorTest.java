package com.pei.dehaze.service.importexport;

import com.alibaba.excel.EasyExcel;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PushbackInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 导入导出文件生成器单元测试
 * <p>覆盖 Excel/CSV 的生成与解析、字段映射、日期格式、表头校验。
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@DisplayName("导入导出文件生成器测试")
class ImportExportFileGeneratorTest {

    private final ImportExportFileGenerator generator = new ImportExportFileGenerator();

    private static List<ExportFieldConfig> sampleExportFields() {
        return List.of(
                ExportFieldConfig.of("username", "用户名", 1),
                ExportFieldConfig.of("nickname", "昵称", 2),
                ExportFieldConfig.of("createTime", "创建时间", 3)
        );
    }

    // ==================== Excel 生成 ====================

    @Test
    @DisplayName("writeExcel - 表头按 fields 顺序写入")
    void testWriteExcel_Headers() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1", "张三", "2026-07-27 10:00:00"))
                : List.of();

        generator.writeExcel(baos, sampleExportFields(), provider);

        List<Map<String, String>> rows = readExcel(baos.toByteArray());
        assertEquals(1, rows.size());
        assertEquals("user1", rows.get(0).get("用户名"));
        assertEquals("张三", rows.get(0).get("昵称"));
        assertEquals("2026-07-27 10:00:00", rows.get(0).get("创建时间"));
    }

    @Test
    @DisplayName("writeExcel - 多批次拉取直至空列表停止")
    void testWriteExcel_MultipleBatches() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = new ExportDataProvider() {
            int callCount = 0;

            @Override
            public List<List<Object>> fetchBatch(int pageNum, int pageSize) {
                callCount++;
                if (callCount > 3) {
                    return List.of();
                }
                return List.of(List.of("user" + callCount, "昵称" + callCount, "2026-01-0" + callCount + " 10:00:00"));
            }
        };

        generator.writeExcel(baos, sampleExportFields(), provider);

        List<Map<String, String>> rows = readExcel(baos.toByteArray());
        assertEquals(3, rows.size());
        assertEquals("user1", rows.get(0).get("用户名"));
        assertEquals("user3", rows.get(2).get("用户名"));
    }

    @Test
    @DisplayName("writeExcel - 空数据只写表头")
    void testWriteExcel_EmptyData() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();

        generator.writeExcel(baos, sampleExportFields(), provider);

        List<Map<String, String>> rows = readExcel(baos.toByteArray());
        assertEquals(0, rows.size());
    }

    // ==================== CSV 生成 ====================

    @Test
    @DisplayName("writeCsv - 包含 BOM 头且按表头顺序写入")
    void testWriteCsv_BomAndHeaders() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1", "张三", "2026-07-27 10:00:00"))
                : List.of();

        generator.writeCsv(baos, sampleExportFields(), provider);

        byte[] bytes = baos.toByteArray();
        assertEquals(0xEF, bytes[0] & 0xFF, "应包含 UTF-8 BOM");
        assertEquals(0xBB, bytes[1] & 0xFF);
        assertEquals(0xBF, bytes[2] & 0xFF);

        List<Map<String, String>> rows = readCsv(bytes);
        assertEquals(1, rows.size());
        assertEquals("user1", rows.get(0).get("用户名"));
        assertEquals("张三", rows.get(0).get("昵称"));
    }

    @Test
    @DisplayName("writeCsv - 多批次拉取")
    void testWriteCsv_MultipleBatches() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = new ExportDataProvider() {
            int callCount = 0;

            @Override
            public List<List<Object>> fetchBatch(int pageNum, int pageSize) {
                callCount++;
                if (callCount > 2) {
                    return List.of();
                }
                return List.of(List.of("user" + callCount));
            }
        };

        generator.writeCsv(baos, List.of(ExportFieldConfig.of("username", "用户名", 1)), provider);

        List<Map<String, String>> rows = readCsv(baos.toByteArray());
        assertEquals(2, rows.size());
        assertEquals("user1", rows.get(0).get("用户名"));
        assertEquals("user2", rows.get(1).get("用户名"));
    }

    // ==================== 模板生成 ====================

    @Test
    @DisplayName("writeTemplateExcel - 包含表头和示例数据")
    void testWriteTemplateExcel_WithSampleData() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        List<ImportFieldConfig> fields = List.of(
                ImportFieldConfig.of("username", "用户名", true),
                ImportFieldConfig.of("nickname", "昵称", false));
        List<Map<String, Object>> sampleData = List.of(
                Map.of("username", "zhangsan", "nickname", "张三"));

        generator.writeTemplateExcel(baos, fields, sampleData);

        List<Map<String, String>> rows = readExcel(baos.toByteArray());
        assertEquals(1, rows.size());
        assertEquals("zhangsan", rows.get(0).get("用户名"));
        assertEquals("张三", rows.get(0).get("昵称"));
    }

    @Test
    @DisplayName("writeTemplateCsv - 示例数据为 null 时不报错")
    void testWriteTemplateCsv_NullSampleData() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        List<ImportFieldConfig> fields = List.of(ImportFieldConfig.of("username", "用户名", true));

        generator.writeTemplateCsv(baos, fields, null);

        List<Map<String, String>> rows = readCsv(baos.toByteArray());
        assertEquals(0, rows.size(), "无示例数据,只应包含表头");
    }

    // ==================== 文件解析 ====================

    @Test
    @DisplayName("parseExcel - 按 label 映射到 field")
    void testParseExcel_FieldMapping() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1", "张三"))
                : List.of();
        generator.writeExcel(baos, List.of(
                ExportFieldConfig.of("username", "用户名", 1),
                ExportFieldConfig.of("nickname", "昵称", 2)
        ), provider);

        List<Map<String, Object>> rows = new ArrayList<>();
        generator.parseExcel(new ByteArrayInputStream(baos.toByteArray()),
                List.of(
                        ImportFieldConfig.of("username", "用户名", true),
                        ImportFieldConfig.of("nickname", "昵称", false)
                ),
                (rowNum, row) -> rows.add(row));

        assertEquals(1, rows.size());
        assertEquals("user1", rows.get(0).get("username"));
        assertEquals("张三", rows.get(0).get("nickname"));
    }

    @Test
    @DisplayName("parseCsv - 按 label 映射到 field")
    void testParseCsv_FieldMapping() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("admin", "管理员"))
                : List.of();
        generator.writeCsv(baos, List.of(
                ExportFieldConfig.of("code", "角色编码", 1),
                ExportFieldConfig.of("name", "角色名称", 2)
        ), provider);

        List<Map<String, Object>> rows = new ArrayList<>();
        generator.parseCsv(new ByteArrayInputStream(baos.toByteArray()),
                List.of(
                        ImportFieldConfig.of("code", "角色编码", true),
                        ImportFieldConfig.of("name", "角色名称", false)
                ),
                (rowNum, row) -> rows.add(row));

        assertEquals(1, rows.size());
        assertEquals("admin", rows.get(0).get("code"));
        assertEquals("管理员", rows.get(0).get("name"));
    }

    @Test
    @DisplayName("parse - 自动识别 xlsx 走 parseExcel")
    void testParse_AutoExcel() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1"))
                : List.of();
        generator.writeExcel(baos,
                List.of(ExportFieldConfig.of("username", "用户名", 1)), provider);

        List<Map<String, Object>> rows = new ArrayList<>();
        generator.parse(new ByteArrayInputStream(baos.toByteArray()), "test.xlsx",
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                (rowNum, row) -> rows.add(row));

        assertEquals(1, rows.size());
    }

    @Test
    @DisplayName("parse - 自动识别 csv 走 parseCsv")
    void testParse_AutoCsv() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1"))
                : List.of();
        generator.writeCsv(baos,
                List.of(ExportFieldConfig.of("username", "用户名", 1)), provider);

        List<Map<String, Object>> rows = new ArrayList<>();
        generator.parse(new ByteArrayInputStream(baos.toByteArray()), "test.csv",
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                (rowNum, row) -> rows.add(row));

        assertEquals(1, rows.size());
    }

    @Test
    @DisplayName("parse - 不支持的文件类型抛 A0701")
    void testParse_UnsupportedType() {
        BusinessException ex = assertThrows(BusinessException.class,
                () -> generator.parse(new ByteArrayInputStream(new byte[0]), "test.txt",
                        List.of(ImportFieldConfig.of("a", "A", false)),
                        (rowNum, row) -> {
                        }));
        assertEquals(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("parse - 文件名为 null 抛 A0701")
    void testParse_NullFileName() {
        BusinessException ex = assertThrows(BusinessException.class,
                () -> generator.parse(new ByteArrayInputStream(new byte[0]), null,
                        List.of(ImportFieldConfig.of("a", "A", false)),
                        (rowNum, row) -> {
                        }));
        assertEquals(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("parse - 多列对齐字段顺序")
    void testParse_FieldOrderPreserved() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("v1", "v2", "v3"))
                : List.of();
        generator.writeCsv(baos, List.of(
                ExportFieldConfig.of("f1", "F1", 1),
                ExportFieldConfig.of("f2", "F2", 2),
                ExportFieldConfig.of("f3", "F3", 3)
        ), provider);

        List<Map<String, Object>> rows = new ArrayList<>();
        generator.parseCsv(new ByteArrayInputStream(baos.toByteArray()),
                List.of(
                        ImportFieldConfig.of("f1", "F1", false),
                        ImportFieldConfig.of("f2", "F2", false),
                        ImportFieldConfig.of("f3", "F3", false)
                ),
                (rowNum, row) -> rows.add(row));

        assertEquals(1, rows.size());
        assertEquals("v1", rows.get(0).get("f1"));
        assertEquals("v2", rows.get(0).get("f2"));
        assertEquals("v3", rows.get(0).get("f3"));
    }

    @Test
    @DisplayName("writeExcel - 隐藏字段不写入表头")
    void testWriteExcel_HiddenFieldSkipped() {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        List<ExportFieldConfig> fields = List.of(
                ExportFieldConfig.builder().field("username").label("用户名").order(1).build(),
                ExportFieldConfig.builder().field("password").label("密码").order(2).hidden(true).build()
        );
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1", "secret"))
                : List.of();

        generator.writeExcel(baos, fields, provider);

        List<Map<String, String>> rows = readExcel(baos.toByteArray());
        assertEquals(1, rows.size());
        assertEquals(1, rows.get(0).size(), "隐藏字段不应出现在表头");
        assertTrue(rows.get(0).containsKey("用户名"));
        assertFalse(rows.get(0).containsKey("密码"));
    }

    @Test
    @DisplayName("parseExcel - Excel 格式错误抛 A0704")
    void testParseExcel_InvalidFormat() {
        BusinessException ex = assertThrows(BusinessException.class,
                () -> generator.parseExcel(new ByteArrayInputStream("not an excel".getBytes(StandardCharsets.UTF_8)),
                        List.of(ImportFieldConfig.of("a", "A", false)),
                        (rowNum, row) -> {
                        }));
        assertEquals(ResultCode.IMPORT_FILE_PARSE_ERROR.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("parseCsv - 表头与模板不一致抛 A0705")
    void testParseCsv_HeaderMismatch() {
        String csv = "wrongHeader\nvalue1\n";
        BusinessException ex = assertThrows(BusinessException.class,
                () -> generator.parseCsv(new ByteArrayInputStream(csv.getBytes(StandardCharsets.UTF_8)),
                        List.of(ImportFieldConfig.of("a", "正确表头", true)),
                        (rowNum, row) -> {
                        }));
        assertEquals(ResultCode.IMPORT_TEMPLATE_MISMATCH.getCode(), ex.getResultCode().getCode());
    }

    // ==================== 工具方法 ====================

    private List<Map<String, String>> readExcel(byte[] bytes) {
        List<Map<String, String>> result = new ArrayList<>();
        EasyExcel.read(new ByteArrayInputStream(bytes), new com.alibaba.excel.read.listener.ReadListener<Map<Integer, String>>() {
            final Map<Integer, String> headers = new LinkedHashMap<>();

            @Override
            public void invokeHead(Map<Integer, com.alibaba.excel.metadata.data.ReadCellData<?>> cellDataMap,
                                   com.alibaba.excel.context.AnalysisContext context) {
                cellDataMap.forEach((k, v) -> headers.put(k, v.getStringValue()));
            }

            @Override
            public void invoke(Map<Integer, String> data, com.alibaba.excel.context.AnalysisContext context) {
                Map<String, String> row = new LinkedHashMap<>();
                for (Map.Entry<Integer, String> entry : headers.entrySet()) {
                    if (data.containsKey(entry.getKey())) {
                        row.put(entry.getValue(), data.get(entry.getKey()));
                    }
                }
                result.add(row);
            }

            @Override
            public void doAfterAllAnalysed(com.alibaba.excel.context.AnalysisContext context) {
            }
        }).sheet().doRead();
        return result;
    }

    private List<Map<String, String>> readCsv(byte[] bytes) throws IOException {
        List<Map<String, String>> result = new ArrayList<>();
        PushbackInputStream pis = new PushbackInputStream(new ByteArrayInputStream(bytes), 3);
        byte[] bom = new byte[3];
        int read = pis.read(bom);
        if (read < 3 || bom[0] != (byte) 0xEF || bom[1] != (byte) 0xBB || bom[2] != (byte) 0xBF) {
            if (read > 0) {
                pis.unread(bom, 0, read);
            }
        }
        try (InputStreamReader reader = new InputStreamReader(pis, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .build()
                     .parse(reader)) {
            for (var record : parser) {
                Map<String, String> row = new LinkedHashMap<>();
                for (var entry : parser.getHeaderMap().entrySet()) {
                    if (entry.getValue() < record.size()) {
                        row.put(entry.getKey(), record.get(entry.getValue()));
                    }
                }
                result.add(row);
            }
        }
        return result;
    }
}
