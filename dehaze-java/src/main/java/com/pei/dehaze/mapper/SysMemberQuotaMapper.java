package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysMemberQuota;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface SysMemberQuotaMapper extends BaseMapper<SysMemberQuota> {

    /**
     * upsert 会员额度：user_id + quota_month 唯一键冲突时复活软删行。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id。
     */
    @Insert("INSERT INTO sys_member_quota (user_id, quota_month, dehaze_quota, dehaze_used, evaluate_quota, evaluate_used, deleted, update_time) " +
            "VALUES (#{userId}, #{quotaMonth}, #{dehazeQuota}, #{dehazeUsed}, #{evaluateQuota}, #{evaluateUsed}, 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), deleted = 0, dehaze_quota = VALUES(dehaze_quota), dehaze_used = VALUES(dehaze_used), " +
            "evaluate_quota = VALUES(evaluate_quota), evaluate_used = VALUES(evaluate_used), update_time = NOW()")
    int upsertByUserAndMonth(@Param("userId") Long userId,
                             @Param("quotaMonth") Integer quotaMonth,
                             @Param("dehazeQuota") Integer dehazeQuota,
                             @Param("dehazeUsed") Integer dehazeUsed,
                             @Param("evaluateQuota") Integer evaluateQuota,
                             @Param("evaluateUsed") Integer evaluateUsed);
}
