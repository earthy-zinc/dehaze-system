package com.pei.dehaze.plugin.easyexcel;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Validator;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.extra.spring.SpringUtil;
import cn.hutool.json.JSONUtil;
import com.alibaba.excel.context.AnalysisContext;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.GenderEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.converter.UserConverter;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.model.vo.UserImportVO;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.List;

/**
 * 用户导入监听器
 * <p>
 *  <a href="https://easyexcel.opensource.alibaba.com/docs/current/quickstart/read#%E6%9C%80%E7%AE%80%E5%8D%95%E7%9A%84%E8%AF%BB%E7%9A%84%E7%9B%91%E5%90%AC%E5%99%A8">最简单的读的监听器</a>
 *
 * @author earthyzinc
 * @since 2022/4/10 20:49
 */
@Slf4j
public class UserImportListener extends MyAnalysisEventListener<UserImportVO> {


    // 有效条数
    private int validCount;

    // 无效条数
    private int invalidCount;

    // 导入返回信息
    StringBuilder msg = new StringBuilder();

    // 部门ID
    private final Long deptId;

    private final SysUserService userService;

    private final PasswordEncoder passwordEncoder;

    private final UserConverter userConverter;

    private final SysRoleService roleService;

    private final SysUserRoleService userRoleService;

    public UserImportListener(Long deptId) {
        this.deptId = deptId;
        this.userService = SpringUtil.getBean(SysUserService.class);
        this.passwordEncoder = SpringUtil.getBean(PasswordEncoder.class);
        this.roleService = SpringUtil.getBean(SysRoleService.class);
        this.userRoleService = SpringUtil.getBean(SysUserRoleService.class);
        this.userConverter = SpringUtil.getBean(UserConverter.class);
    }

    /**
     * 每一条数据解析都会来调用
     * <p>
     * 1. 数据校验；全字段校验
     * 2. 数据持久化；
     *
     * @param userImportVO    一行数据，类似于 {@link AnalysisContext#readRowHolder()}
     */
    @Override
    public void invoke(UserImportVO userImportVO, AnalysisContext analysisContext) {
        log.info("解析到一条用户数据:{}", JSONUtil.toJsonStr(userImportVO));
        StringBuilder validationMsg = validateUser(userImportVO);

        if (validationMsg.isEmpty()) {
            // 校验通过，持久化至数据库
            SysUser entity = userConverter.importVo2Entity(userImportVO);
            entity.setDeptId(deptId);   // 部门
            entity.setPassword(passwordEncoder.encode(SystemConstants.DEFAULT_PASSWORD));   // 默认密码
            // 性别翻译
            String genderLabel = userImportVO.getGenderLabel();
            if (CharSequenceUtil.isNotBlank(genderLabel)) {
                Integer genderValue = (Integer) IBaseEnum.getValueByLabel(genderLabel, GenderEnum.class);
                entity.setGender(genderValue);
            }

            // 角色解析
            String roleCodes = userImportVO.getRoleCodes();
            List<Long> roleIds = null;
            if (CharSequenceUtil.isNotBlank(roleCodes)) {
                roleIds = roleService.list(
                                new LambdaQueryWrapper<SysRole>()
                                        .in(SysRole::getCode,
                                        (Object[]) roleCodes.split(","))
                                        .eq(SysRole::getStatus, StatusEnum.ENABLE.getValue())
                                        .select(SysRole::getId)
                        ).stream()
                        .map(SysRole::getId)
                        .toList();
            }


            boolean saveResult = userService.save(entity);
            if (saveResult) {
                validCount++;
                // 保存用户角色关联
                if (CollUtil.isNotEmpty(roleIds)) {
                    List<SysUserRole> userRoles = roleIds.stream()
                            .map(roleId -> new SysUserRole(entity.getId(), roleId))
                            .toList();
                    userRoleService.saveBatch(userRoles);
                }
            } else {
                invalidCount++;
                msg.append("第").append(validCount + invalidCount).append("行数据保存失败；<br/>");
            }
        } else {
            invalidCount++;
            msg
                    .append("第")
                    .append(validCount + invalidCount)
                    .append("行数据校验失败：")
                    .append(validationMsg)
                    .append("<br/>");
        }
    }

    @NotNull
    private StringBuilder validateUser(UserImportVO userImportVO) {
        // 校验数据
        StringBuilder validationMsg = new StringBuilder();

        String username = userImportVO.getUsername();
        if (CharSequenceUtil.isBlank(username)) {
            validationMsg.append("用户名为空；");
        } else {
            long count = userService.count(new LambdaQueryWrapper<SysUser>().eq(SysUser::getUsername, username));
            if (count > 0) {
                validationMsg.append("用户名已存在；");
            }
        }

        String nickname = userImportVO.getNickname();
        if (CharSequenceUtil.isBlank(nickname)) {
            validationMsg.append("用户昵称为空；");
        }

        String mobile = userImportVO.getMobile();
        if (CharSequenceUtil.isBlank(mobile)) {
            validationMsg.append("手机号码为空；");
        } else {
            if (!Validator.isMobile(mobile)) {
                validationMsg.append("手机号码不正确；");
            }
        }
        return validationMsg;
    }


    /**
     * 所有数据解析完成会来调用
     */
    @Override
    public void doAfterAllAnalysed(AnalysisContext analysisContext) {
        // 暂时不需要
    }


    @Override
    public String getMsg() {
        // 总结信息
        return CharSequenceUtil.format(
                "导入用户结束：成功{}条，失败{}条；<br/>{}",
                validCount, invalidCount, msg);
    }
}
