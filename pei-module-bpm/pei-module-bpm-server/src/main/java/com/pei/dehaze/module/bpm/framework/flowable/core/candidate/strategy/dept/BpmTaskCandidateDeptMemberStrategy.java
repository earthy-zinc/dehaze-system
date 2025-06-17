package com.pei.dehaze.module.bpm.framework.flowable.core.candidate.strategy.dept;

import com.pei.dehaze.framework.common.util.string.StrUtils;
import com.pei.dehaze.module.bpm.framework.flowable.core.candidate.BpmTaskCandidateStrategy;
import com.pei.dehaze.module.bpm.framework.flowable.core.enums.BpmTaskCandidateStrategyEnum;
import com.pei.dehaze.module.system.api.dept.DeptApi;
import com.pei.dehaze.module.system.api.user.AdminUserApi;
import com.pei.dehaze.module.system.api.user.dto.AdminUserRespDTO;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;

import static com.pei.dehaze.framework.common.util.collection.CollectionUtils.convertSet;

/**
 * 部门的成员 {@link BpmTaskCandidateStrategy} 实现类
 *
 * @author kyle
 */
@Component
public class BpmTaskCandidateDeptMemberStrategy implements BpmTaskCandidateStrategy {

    @Resource
    private DeptApi deptApi;
    @Resource
    private AdminUserApi adminUserApi;

    @Override
    public BpmTaskCandidateStrategyEnum getStrategy() {
        return BpmTaskCandidateStrategyEnum.DEPT_MEMBER;
    }

    @Override
    public void validateParam(String param) {
        Set<Long> deptIds = StrUtils.splitToLongSet(param);
        deptApi.validateDeptList(deptIds).checkError();
    }

    @Override
    public Set<Long> calculateUsers(String param) {
        Set<Long> deptIds = StrUtils.splitToLongSet(param);
        List<AdminUserRespDTO> users = adminUserApi.getUserListByDeptIds(deptIds).getCheckedData();
        return convertSet(users, AdminUserRespDTO::getId);
    }

}
