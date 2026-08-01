package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysMember;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysMemberMapper extends BaseMapper<SysMember> {

    /**
     * upsert 会员：user_id 唯一键冲突时复活软删行。
     * 安全原因：user_id 相同即同一自然人，无越权风险。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id。
     * 业务决策：复活时降级为 level_0、清空 monthly_*_quota，保留 total_consumption（风控用）。
     */
    @Insert("INSERT INTO sys_member (user_id, level_code, total_consumption, deleted, update_time) " +
            "VALUES (#{userId}, 'level_0', #{totalConsumption}, 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), deleted = 0, level_code = VALUES(level_code), " +
            "total_consumption = VALUES(total_consumption), update_time = NOW()")
    int upsertByUser(@Param("userId") Long userId,
                     @Param("totalConsumption") Long totalConsumption);
}
