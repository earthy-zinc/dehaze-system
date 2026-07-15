package com.pei.dehaze.plugin.mybatis.handler;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.toolkit.StringPool;
import com.baomidou.mybatisplus.extension.plugins.handler.DataPermissionHandler;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.DataScopeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.plugin.mybatis.annotation.DataPermission;
import com.pei.dehaze.security.util.SecurityUtils;
import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.expression.Function;
import net.sf.jsqlparser.expression.LongValue;
import net.sf.jsqlparser.expression.operators.conditional.AndExpression;
import net.sf.jsqlparser.expression.operators.conditional.OrExpression;
import net.sf.jsqlparser.expression.operators.relational.EqualsTo;
import net.sf.jsqlparser.expression.operators.relational.ExpressionList;
import net.sf.jsqlparser.expression.operators.relational.InExpression;
import net.sf.jsqlparser.schema.Column;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.select.PlainSelect;
import net.sf.jsqlparser.statement.select.SelectExpressionItem;
import net.sf.jsqlparser.statement.select.SubSelect;

import java.lang.reflect.Method;
import java.util.List;

/**
 * 数据权限控制器
 *
 * @author zc
 * @since 2021-12-10 13:28
 */
public class MyDataPermissionHandler implements DataPermissionHandler {

    @Override
    public Expression getSqlSegment(Expression where, String mappedStatementId) {
        Class<?> clazz;
        try {
            clazz = Class.forName(mappedStatementId.substring(0, mappedStatementId.lastIndexOf(StringPool.DOT)));
        } catch (ClassNotFoundException e) {
            throw new BusinessException("数据权限处理器找不到Mapper类: " + mappedStatementId, e);
        }
        String methodName = mappedStatementId.substring(mappedStatementId.lastIndexOf(StringPool.DOT) + 1);
        Method[] methods = clazz.getDeclaredMethods();
        for (Method method : methods) {
            if (method.getName().equals(methodName)) {
                DataPermission annotation = method.getAnnotation(DataPermission.class);
                // 如果没有注解或者是超级管理员，直接返回
                if (annotation == null || SecurityUtils.isRoot()) {
                    return where;
                }
                return dataScopeFilter(annotation.deptAlias(), annotation.deptIdColumnName(), annotation.userAlias(), annotation.userIdColumnName(), where);
            }
        }
        return where;
    }

    /**
     * 构建过滤条件
     *
     * @param where 当前查询条件
     * @return 构建后查询条件
     */
    public static Expression dataScopeFilter(String deptAlias, String deptIdColumnName, String userAlias, String userIdColumnName, Expression where) {
        String deptColumnName = CharSequenceUtil.isNotBlank(deptAlias) ? (deptAlias + StringPool.DOT + deptIdColumnName) : deptIdColumnName;
        String userColumnName = CharSequenceUtil.isNotBlank(userAlias) ? (userAlias + StringPool.DOT + userIdColumnName) : userIdColumnName;

        // 获取当前用户的数据权限
        Integer dataScope = SecurityUtils.getDataScope();
        if (dataScope == null) {
            return where;
        }

        DataScopeEnum dataScopeEnum = IBaseEnum.getEnumByValue(dataScope, DataScopeEnum.class);

        Expression appendExpression;
        switch (dataScopeEnum) {
            case ALL:
                return where;
            case DEPT:
                appendExpression = new EqualsTo(new Column(deptColumnName), new LongValue(SecurityUtils.getDeptId()));
                break;
            case SELF:
                appendExpression = new EqualsTo(new Column(userColumnName), new LongValue(SecurityUtils.getUserId()));
                break;
            // 默认部门及子部门数据权限
            default:
                appendExpression = buildDeptAndSubFilter(deptColumnName, SecurityUtils.getDeptId());
                break;
        }

        if (where == null) {
            return appendExpression;
        }

        return new AndExpression(where, appendExpression);
    }

    /**
     * 构建部门及子部门数据权限过滤条件
     * <p>
     * SQL: deptColumnName IN (SELECT id FROM sys_dept WHERE id = deptId OR FIND_IN_SET(deptId, tree_path))
     */
    private static Expression buildDeptAndSubFilter(String deptColumnName, long deptId) {
        PlainSelect plainSelect = new PlainSelect();
        plainSelect.addSelectItems(new SelectExpressionItem(new Column("id")));
        plainSelect.setFromItem(new Table("sys_dept"));
        plainSelect.setWhere(new OrExpression(
                new EqualsTo(new Column("id"), new LongValue(deptId)),
                buildFindInSet(deptId, "tree_path")
        ));

        SubSelect subSelect = new SubSelect();
        subSelect.setSelectBody(plainSelect);

        return new InExpression(new Column(deptColumnName), subSelect);
    }

    /**
     * 构建 FIND_IN_SET(value, columnName) 函数表达式
     */
    private static Function buildFindInSet(long value, String columnName) {
        Function findInSet = new Function();
        findInSet.setName("FIND_IN_SET");
        findInSet.setParameters(new ExpressionList(List.of(new LongValue(value), new Column(columnName))));
        return findInSet;
    }
}
