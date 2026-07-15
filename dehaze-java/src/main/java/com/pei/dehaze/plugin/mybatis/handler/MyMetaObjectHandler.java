package com.pei.dehaze.plugin.mybatis.handler;

import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.security.util.SecurityUtils;
import org.apache.ibatis.reflection.MetaObject;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * mybatis-plus 字段自动填充
 *
 * @author earthyzinc
 * @since 2022/10/14
 */
@Component
public class MyMetaObjectHandler implements MetaObjectHandler {

    /**
     * 新增填充创建时间
     *
     * @param metaObject 元数据
     */
    @Override
    public void insertFill(MetaObject metaObject) {
        this.strictInsertFill(metaObject, "createTime", LocalDateTime::now, LocalDateTime.class);
        this.strictUpdateFill(metaObject, "updateTime", LocalDateTime::now, LocalDateTime.class);
        this.strictInsertFill(metaObject, "createBy", this::currentUserId, Long.class);
        this.strictUpdateFill(metaObject, "updateBy", this::currentUserId, Long.class);
    }

    /**
     * 更新填充更新时间
     *
     * @param metaObject 元数据
     */
    @Override
    public void updateFill(MetaObject metaObject) {
        this.strictUpdateFill(metaObject, "updateTime", LocalDateTime::now, LocalDateTime.class);
        this.strictUpdateFill(metaObject, "updateBy", this::currentUserId, Long.class);
    }

    /**
     * 获取当前操作用户ID，无登录上下文时回退为系统用户ID
     */
    private Long currentUserId() {
        Long userId = SecurityUtils.getUserId();
        return userId != null ? userId : SystemConstants.SYSTEM_USER_ID;
    }

}
