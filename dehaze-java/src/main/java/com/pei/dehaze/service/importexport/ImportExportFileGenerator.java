package com.pei.dehaze.service.importexport;

import com.alibaba.excel.EasyExcel;
import com.alibaba.excel.ExcelWriter;
import com.alibaba.excel.context.AnalysisContext;
import com.alibaba.excel.read.listener.ReadListener;
import com.alibaba.excel.write.metadata.WriteSheet;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.InputStreamReader;
import java.io.PushbackInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 导入导出文件生成器
 * <p>封装 EasyExcel 和 Apache Commons CSV 的流式写入/读取逻辑，
 * 提供 Excel/CSV 的统一生成和解析能力。
 */
@Slf4j
@Component
public class ImportExportFileGenerator {

    /**
     * 流式写入 Excel（分批拉取数据，避免内存溢出）
     */
    public void writeExcel(OutputStream os, List<ExportFieldConfig> fields, ExportDataProvider dataProvider) {
        List<List<String>> head = buildExcelHead(fields);
        try (ExcelWriter writer = EasyExcel.write(os).head(head).build()) {
            WriteSheet sheet = EasyExcel.writerSheet("Sheet1").build();
            int pageNum = 1;
            boolean hasData = false;
            while (true) {
                List<List<Object>> batch = dataProvider.fetchBatch(pageNum, 1000);
                if (batch == null || batch.isEmpty()) {
                    break;
                }
                hasData = true;
                List<List<Object>> data = batch.stream()
                        .map(row -> alignRow(row, fields))
                        .collect(Collectors.toList());
                writer.write(data, sheet);
                pageNum++;
            }
            if (!hasData) {
                writer.write(List.of(), sheet);
            }
        }
    }

    /**
     * 流式写入 CSV
     */
    public void writeCsv(OutputStream os, List<ExportFieldConfig> fields, ExportDataProvider dataProvider) throws IOException {
        os.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
        String[] headers = fields.stream().map(ExportFieldConfig::getLabel).toArray(String[]::new);
        try (OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(osw, CSVFormat.DEFAULT.withHeader(headers))) {
            int pageNum = 1;
            while (true) {
                List<List<Object>> batch = dataProvider.fetchBatch(pageNum, 1000);
                if (batch == null || batch.isEmpty()) {
                    break;
                }
                for (List<Object> row : batch) {
                    printer.printRecord(row);
                }
                pageNum++;
            }
        }
    }

    /**
     * 生成导入模板 Excel（含表头和示例数据）
     */
    public void writeTemplateExcel(OutputStream os, List<ImportFieldConfig> fields,
                                   List<Map<String, Object>> sampleData) {
        List<List<String>> head = fields.stream()
                .map(f -> {
                    List<String> h = new ArrayList<>(2);
                    h.add(f.getLabel());
                    return h;
                })
                .collect(Collectors.toList());

        List<List<Object>> data = new ArrayList<>();
        if (sampleData != null) {
            for (Map<String, Object> row : sampleData) {
                List<Object> rowData = new ArrayList<>(fields.size());
                for (ImportFieldConfig f : fields) {
                    Object v = row.get(f.getField());
                    rowData.add(v == null ? "" : v);
                }
                data.add(rowData);
            }
        }

        try (ExcelWriter writer = EasyExcel.write(os).head(head).build()) {
            WriteSheet sheet = EasyExcel.writerSheet("导入模板").build();
            writer.write(data, sheet);
        }
    }

    /**
     * 生成导入模板 CSV
     */
    public void writeTemplateCsv(OutputStream os, List<ImportFieldConfig> fields,
                                 List<Map<String, Object>> sampleData) throws IOException {
        os.write(new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF});
        String[] headers = fields.stream().map(ImportFieldConfig::getLabel).toArray(String[]::new);
        try (OutputStreamWriter osw = new OutputStreamWriter(os, StandardCharsets.UTF_8);
             CSVPrinter printer = new CSVPrinter(osw, CSVFormat.DEFAULT.builder().setHeader(headers).build())) {
            if (sampleData != null) {
                for (Map<String, Object> row : sampleData) {
                    List<Object> record = new ArrayList<>(fields.size());
                    for (ImportFieldConfig f : fields) {
                        Object v = row.get(f.getField());
                        record.add(v == null ? "" : v);
                    }
                    printer.printRecord(record);
                }
            }
        }
    }

    /**
     * 解析 Excel 文件为数据行（流式读取，避免内存溢出）
     * @param is          输入流
     * @param fields      字段配置（用于表头映射）
     * @param rowConsumer 每行数据回调
     */
    public void parseExcel(InputStream is, List<ImportFieldConfig> fields, RowConsumer rowConsumer) {
        Map<String, ImportFieldConfig> labelToField = new LinkedHashMap<>();
        for (ImportFieldConfig f : fields) {
            labelToField.put(f.getLabel(), f);
        }

        try {
            EasyExcel.read(is, new ReadListener<Map<Integer, String>>() {
                private final Map<Integer, String> headers = new LinkedHashMap<>();
                private int rowNum = 0;

                @Override
                public void invokeHead(Map<Integer, com.alibaba.excel.metadata.data.ReadCellData<?>> cellDataMap,
                                       AnalysisContext context) {
                    cellDataMap.forEach((k, v) -> headers.put(k, v.getStringValue()));
                }

                @Override
                public void invoke(Map<Integer, String> data, AnalysisContext context) {
                    rowNum++;
                    Map<String, Object> row = new LinkedHashMap<>();
                    for (Map.Entry<Integer, String> entry : headers.entrySet()) {
                        String label = entry.getValue();
                        ImportFieldConfig fc = labelToField.get(label);
                        if (fc != null && data.containsKey(entry.getKey())) {
                            row.put(fc.getField(), data.get(entry.getKey()));
                        }
                    }
                    rowConsumer.consume(rowNum, row);
                }

                @Override
                public void doAfterAllAnalysed(AnalysisContext context) {
                }
            }).sheet().doRead();
        } catch (Exception e) {
            log.error("Excel 解析失败", e);
            throw new BusinessException(ResultCode.IMPORT_FILE_PARSE_ERROR, "Excel 文件解析失败: " + e.getMessage());
        }
    }

    /**
     * 解析 CSV 文件为数据行
     */
    public void parseCsv(InputStream is, List<ImportFieldConfig> fields, RowConsumer rowConsumer) throws IOException {
        Map<String, ImportFieldConfig> labelToField = new LinkedHashMap<>();
        Set<String> requiredLabels = new HashSet<>();
        for (ImportFieldConfig f : fields) {
            labelToField.put(f.getLabel(), f);
            if (f.isRequired()) {
                requiredLabels.add(f.getLabel());
            }
        }

        PushbackInputStream pis = new PushbackInputStream(is, 3);
        byte[] bom = new byte[3];
        int read = pis.read(bom);
        if (read < 3 || bom[0] != (byte) 0xEF || bom[1] != (byte) 0xBB || bom[2] != (byte) 0xBF) {
            if (read > 0) {
                pis.unread(bom, 0, read);
            }
        }

        try (InputStreamReader reader = new InputStreamReader(pis, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader((String[]) null)
                     .setSkipHeaderRecord(true)
                     .build()
                     .parse(reader)) {

            Map<String, Integer> headerMap = parser.getHeaderMap();
            Set<String> csvHeaders = headerMap.keySet();
            for (String required : requiredLabels) {
                if (!csvHeaders.contains(required)) {
                    throw new BusinessException(ResultCode.IMPORT_TEMPLATE_MISMATCH,
                            "CSV 缺少必填表头: " + required);
                }
            }

            int rowNum = 0;
            for (CSVRecord record : parser) {
                rowNum++;
                Map<String, Object> row = new LinkedHashMap<>();
                for (Map.Entry<String, Integer> entry : headerMap.entrySet()) {
                    String label = entry.getKey();
                    ImportFieldConfig fc = labelToField.get(label);
                    if (fc != null && entry.getValue() < record.size()) {
                        row.put(fc.getField(), record.get(entry.getValue()));
                    }
                }
                rowConsumer.consume(rowNum, row);
            }
        }
    }

    /**
     * 从输入流解析文件（自动识别 Excel/CSV）
     */
    public void parse(InputStream is, String fileName, List<ImportFieldConfig> fields, RowConsumer rowConsumer) throws IOException {
        String lower = fileName == null ? "" : fileName.toLowerCase();
        if (lower.endsWith(".csv")) {
            parseCsv(is, fields, rowConsumer);
        } else if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) {
            parseExcel(is, fields, rowConsumer);
        } else {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH,
                    "不支持的文件类型: " + fileName);
        }
    }

    private List<List<String>> buildExcelHead(List<ExportFieldConfig> fields) {
        return fields.stream()
                .filter(f -> !f.isHidden())
                .sorted((a, b) -> Integer.compare(a.getOrder(), b.getOrder()))
                .map(f -> {
                    List<String> h = new ArrayList<>(1);
                    h.add(f.getLabel());
                    return h;
                })
                .collect(Collectors.toList());
    }

    private List<Object> alignRow(List<Object> row, List<ExportFieldConfig> fields) {
        List<Object> aligned = new ArrayList<>(fields.size());
        for (int i = 0; i < fields.size(); i++) {
            aligned.add(i < row.size() ? row.get(i) : "");
        }
        return aligned;
    }

    /**
     * 行数据消费回调
     */
    @FunctionalInterface
    public interface RowConsumer {
        void consume(int rowNum, Map<String, Object> row);
    }
}
