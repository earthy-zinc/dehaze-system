package com.pei.dehaze.module.report.service.goview;

import com.google.common.collect.Maps;
import com.pei.dehaze.module.report.controller.admin.goview.vo.data.GoViewDataRespVO;
import jakarta.annotation.Resource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.rowset.SqlRowSet;
import org.springframework.jdbc.support.rowset.SqlRowSetMetaData;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * GoView 数据 Service 实现类
 * <p>
 * 补充说明： 1. 目前默认使用 jdbcTemplate 查询项目配置的数据源。如果你想查询其它数据源，可以新建对应数据源的 jdbcTemplate
 * 来实现。 2. 默认数据源是 MySQL
 * 关系数据源，可能数据量比较大的情况下，会比较慢，可以考虑后续使用 Click House 等等。
 *
 * @author earthyzinc
 */
@Service
@Validated
public class GoViewDataServiceImpl implements GoViewDataService {

    @Resource
    private JdbcTemplate jdbcTemplate;

    @Override
    public GoViewDataRespVO getDataBySQL(String sql) {
        // 0. SQL 安全校验，防止 SQL 注入
        validateSQL(sql);

        // 1. 执行查询
        SqlRowSet sqlRowSet = jdbcTemplate.queryForRowSet(sql);

        // 2. 构建返回结果
        GoViewDataRespVO respVO = new GoViewDataRespVO();
        // 2.1 解析元数据
        SqlRowSetMetaData metaData = sqlRowSet.getMetaData();
        String[] columnNames = metaData.getColumnNames();
        respVO.setDimensions(Arrays.asList(columnNames));
        // 2.2 解析数据明细
        respVO.setSource(new LinkedList<>()); // 由于数据量不确认，使用 LinkedList 虽然内存占用大一点，但是不存在扩容复制的问题
        while (sqlRowSet.next()) {
            Map<String, Object> data = Maps.newHashMapWithExpectedSize(columnNames.length);
            for (String columnName : columnNames) {
                data.put(columnName, sqlRowSet.getObject(columnName));
            }
            respVO.getSource().add(data);
        }
        return respVO;
    }

    /**
     * SQL 安全校验：仅允许 SELECT 查询，禁止危险操作
     */
    private void validateSQL(String sql) {
        if (sql == null || sql.isBlank()) {
            throw new IllegalArgumentException("SQL 语句不能为空");
        }
        String trimmedSql = sql.trim();

        // 禁止多语句执行（分号分隔的多条SQL）
        // 移除字符串末尾可能存在的单个分号后，检查是否还包含分号
        String sqlWithoutTrailingSemicolon = trimmedSql.replaceAll(";\\s*$", "");
        if (sqlWithoutTrailingSemicolon.contains(";")) {
            throw new IllegalArgumentException("不允许执行多条 SQL 语句");
        }

        // 仅允许 SELECT 语句
        if (!Pattern.compile("^\\s*SELECT\\b", Pattern.CASE_INSENSITIVE).matcher(trimmedSql).find()) {
            throw new IllegalArgumentException("仅允许执行 SELECT 查询语句");
        }

        // 禁止危险关键字（独立单词匹配，避免误伤字段名）
        String[] dangerousKeywords = {
                "INSERT\\s+INTO", "UPDATE\\s+\\S+\\s+SET", "DELETE\\s+FROM",
                "DROP\\s+", "ALTER\\s+", "TRUNCATE\\s+", "CREATE\\s+",
                "EXEC\\s*\\(", "EXECUTE\\s+", "CALL\\s+",
                "GRANT\\s+", "REVOKE\\s+",
                "UNION\\s+(ALL\\s+)?SELECT",
                "INTO\\s+OUTFILE", "INTO\\s+DUMPFILE",
                "LOAD_FILE\\s*\\(", "LOAD\\s+DATA"
        };
        for (String keyword : dangerousKeywords) {
            if (Pattern.compile(keyword, Pattern.CASE_INSENSITIVE).matcher(trimmedSql).find()) {
                throw new IllegalArgumentException("SQL 语句包含不允许的危险操作: " + keyword.replaceAll("\\\\[sSb+*?()]+", ""));
            }
        }

        // 禁止注释语法（常被用于绕过安全检查）
        if (trimmedSql.contains("--") || trimmedSql.contains("/*")) {
            throw new IllegalArgumentException("SQL 语句不允许包含注释");
        }
    }

}
