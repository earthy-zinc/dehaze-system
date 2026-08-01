package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAutoRenew;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysAutoRenewMapper extends BaseMapper<SysAutoRenew> {

    /**
     * upsert 自动续费：user_id + package_id 唯一键冲突时复活软删行。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id。
     */
    @Insert("INSERT INTO sys_auto_renew (user_id, package_id, pay_method, status, fail_count, deleted, update_time) " +
            "VALUES (#{userId}, #{packageId}, #{payMethod}, #{status}, #{failCount}, 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), deleted = 0, pay_method = VALUES(pay_method), " +
            "status = VALUES(status), fail_count = VALUES(fail_count), update_time = NOW()")
    int upsertByUserAndPackage(
            @Param("userId") Long userId,
            @Param("packageId") Long packageId,
            @Param("payMethod") String payMethod,
            @Param("status") Integer status,
            @Param("failCount") Integer failCount);
}
