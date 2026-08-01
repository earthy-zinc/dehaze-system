package com.pei.dehaze.common.handler;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;
import org.apache.ibatis.type.MappedTypes;

import java.sql.CallableStatement;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collections;
import java.util.List;

/**
 * List&lt;Long&gt; 的 JSON 列类型处理器。
 *
 * <p>MyBatis-Plus 自带的 {@link com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler}
 * 通过反射获取字段类型，但 Java 泛型擦除导致 {@code List<Long>} 在运行时退化为 raw {@code List}，
 * Jackson 反序列化 JSON 数字时默认用 Integer 接收，序列化输出时 LongSerializer 强转 Integer→Long 报错。
 *
 * <p>本处理器在反序列化时显式传入 {@link TypeReference}，让 Jackson 知道元素类型为 Long，
 * 从根本上解决类型转换异常。
 */
@MappedTypes(List.class)
public class LongListTypeHandler extends BaseTypeHandler<List<Long>> {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final TypeReference<List<Long>> TYPE_REF = new TypeReference<>() {};

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, List<Long> parameter, JdbcType jdbcType) throws SQLException {
        try {
            ps.setString(i, MAPPER.writeValueAsString(parameter));
        } catch (JsonProcessingException e) {
            throw new SQLException("序列化 List<Long> 失败", e);
        }
    }

    @Override
    public List<Long> getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return parse(rs.getString(columnName));
    }

    @Override
    public List<Long> getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return parse(rs.getString(columnIndex));
    }

    @Override
    public List<Long> getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return parse(cs.getString(columnIndex));
    }

    private List<Long> parse(String json) {
        if (json == null || json.isBlank()) {
            return Collections.emptyList();
        }
        try {
            return MAPPER.readValue(json, TYPE_REF);
        } catch (JsonProcessingException e) {
            return Collections.emptyList();
        }
    }
}
