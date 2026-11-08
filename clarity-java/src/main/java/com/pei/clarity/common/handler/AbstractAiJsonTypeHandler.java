package com.pei.clarity.common.handler;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.sql.CallableStatement;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;

/**
 * AI 契约 JSON 列类型处理器基类。
 *
 * <p>MyBatis-Plus 自带的 {@link com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler}
 * 只接受 {@code Class}，泛型擦除后 {@code List<Block>} 会退化为 {@code List<Map>}；本基类由子类显式传入
 * {@link TypeReference}，保证 {@code blocks / plan / pending_interrupt} 按契约 DTO 读写。
 *
 * <p>解析失败直接抛 {@link SQLException}：这些列由契约写入方按 Schema 序列化，反序列化失败即为数据损坏，
 * 静默降级成空值会让前端把损坏消息渲染成空消息，掩盖问题。
 */
public abstract class AbstractAiJsonTypeHandler<T> extends BaseTypeHandler<T> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final TypeReference<T> typeReference;

    protected AbstractAiJsonTypeHandler(TypeReference<T> typeReference) {
        this.typeReference = typeReference;
    }

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, T parameter, JdbcType jdbcType) throws SQLException {
        try {
            ps.setString(i, MAPPER.writeValueAsString(parameter));
        } catch (JsonProcessingException e) {
            throw new SQLException("序列化 AI 契约 JSON 失败", e);
        }
    }

    @Override
    public T getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return parse(rs.getString(columnName));
    }

    @Override
    public T getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return parse(rs.getString(columnIndex));
    }

    @Override
    public T getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return parse(cs.getString(columnIndex));
    }

    private T parse(String json) throws SQLException {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return MAPPER.readValue(json, typeReference);
        } catch (JsonProcessingException e) {
            throw new SQLException("反序列化 AI 契约 JSON 失败", e);
        }
    }
}
