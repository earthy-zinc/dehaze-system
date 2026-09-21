package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.config.GlobalConfig;
import com.baomidou.mybatisplus.core.metadata.TableFieldInfo;
import com.baomidou.mybatisplus.core.metadata.TableInfo;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 唯一键感知逻辑删除（方案A）：deleted 列存"删除时的行id"。
 * 验证 @TableLogic(value = "0", delval = "id") 在项目 MP 版本下生成的 SQL 片段
 * 为 deleted=0（读路径）与 deleted=id（删除时写行id），保证软删行不占唯一键位。
 */
class TableLogicDelvalTest {

    private static TableInfo tableInfo;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfig globalConfig = GlobalConfigUtils.defaults();
        globalConfig.getDbConfig().setLogicDeleteField("deleted");
        GlobalConfigUtils.setGlobalConfig(configuration, globalConfig);

        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "");
        assistant.setCurrentNamespace("com.pei.dehaze.mapper.SysDictTypeMapper");
        tableInfo = TableInfoHelper.initTableInfo(assistant, SysDictType.class);
    }

    @Test
    void notDeleteConditionShouldBeZero() {
        // isWhere=true 时取 logicNotDeleteValue：所有读路径条件 deleted=0，语义不变
        assertEquals("deleted=0", tableInfo.getLogicDeleteSql(false, true));
    }

    @Test
    void deleteValueShouldReferenceIdColumn() {
        // isWhere=false 时取 delval="id"，MP 将其原样拼接进 SQL（非参数绑定），
        // 生成 UPDATE ... SET deleted=id，即删除时写入行id
        assertEquals("deleted=id", tableInfo.getLogicDeleteSql(false, false));
    }

    @Test
    void logicDeleteFieldShouldBeNonCharSequenceLong() {
        // delval 原样拼接依赖字段非 CharSequence（String 会生成 deleted='id'，故 deleted 必须保持数值类型）
        List<TableFieldInfo> logicFields = Optional.of(tableInfo.getFieldList())
                .orElseThrow()
                .stream()
                .filter(TableFieldInfo::isLogicDelete)
                .toList();
        assertEquals(1, logicFields.size());
        TableFieldInfo field = logicFields.get(0);
        assertEquals("deleted", field.getColumn());
        assertEquals(Long.class, field.getPropertyType());
        assertFalse(field.isCharSequence());
        assertTrue(field.getPropertyType() instanceof Class<?> && Number.class.isAssignableFrom(field.getPropertyType()));
    }
}
