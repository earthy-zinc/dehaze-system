package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Validator;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.GenderEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import com.pei.dehaze.service.importexport.ImportHandler;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 用户导入处理器
 * <p>支持全量/部分两种模式：
 * <ul>
 *   <li>全量模式(all)：任意行校验失败则整体回滚</li>
 *   <li>部分模式(partial)：跳过校验失败行，继续导入有效数据</li>
 * </ul>
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class UserImportHandler implements ImportHandler {

    private final SysUserService userService;
    private final SysRoleService roleService;
    private final SysUserRoleService userRoleService;
    private final PasswordEncoder passwordEncoder;

    /**
     * 新用户默认密码（由各 profile 的 system.default-password 注入，源为根 .env 的 DEFAULT_PASSWORD）
     */
    @Value("${system.default-password}")
    private String defaultPassword;

    @Override
    public String getModule() {
        return "user";
    }

    @Override
    public List<ImportFieldConfig> getFieldConfigs() {
        return List.of(
                ImportFieldConfig.builder().field("username").label("用户名").required(true).maxLength(64).build(),
                ImportFieldConfig.builder().field("nickname").label("昵称").required(true).maxLength(50).build(),
                ImportFieldConfig.builder().field("genderLabel").label("性别").required(false).build(),
                ImportFieldConfig.builder().field("mobile").label("手机号").required(true).regex("^1[3-9]\\d{9}$").build(),
                ImportFieldConfig.builder().field("email").label("邮箱").required(false).regex("^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$").build(),
                ImportFieldConfig.builder().field("roleCodes").label("角色编码(多个逗号分隔)").required(false).build()
        );
    }

    @Override
    public List<Map<String, Object>> getTemplateSampleData() {
        Map<String, Object> sample = new LinkedHashMap<>();
        sample.put("username", "zhangsan");
        sample.put("nickname", "张三");
        sample.put("genderLabel", "男");
        sample.put("mobile", "13800138000");
        sample.put("email", "zhangsan@example.com");
        sample.put("roleCodes", "user");
        return List.of(sample);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ImportResult importBatch(List<Map<String, Object>> rows, ImportOptions options, ProgressCallback callback) {
        boolean partial = options.isPartialMode();
        Long deptId = extractLong(options.getExtraParams(), "deptId");
        String defaultEncryptPwd = passwordEncoder.encode(defaultPassword);

        List<ImportResult.ImportError> errors = new ArrayList<>();
        int successCount = 0;
        int failureCount = 0;
        int total = rows.size();

        // 预加载所有角色编码->ID 映射（避免逐行查询）
        Map<String, Long> roleCodeMap = loadRoleCodeMap(rows);

        for (int i = 0; i < total; i++) {
            int rowNum = i + 2; // 表头占第1行
            Map<String, Object> row = rows.get(i);
            callback.updateProgress(i + 1, total, "导入第 " + rowNum + " 行");

            try {
                String validationError = validateRow(row);
                if (validationError != null) {
                    throw new IllegalArgumentException(validationError);
                }

                SysUser entity = buildEntity(row, deptId, defaultEncryptPwd, roleCodeMap);
                boolean saved = userService.save(entity);
                if (!saved) {
                    throw new IllegalStateException("保存失败");
                }

                List<Long> roleIds = parseRoleIds(row, roleCodeMap);
                if (CollUtil.isNotEmpty(roleIds)) {
                    List<SysUserRole> userRoles = roleIds.stream()
                            .map(rid -> new SysUserRole(entity.getId(), rid))
                            .toList();
                    userRoleService.saveBatch(userRoles);
                }
                successCount++;
            } catch (Exception e) {
                failureCount++;
                errors.add(ImportResult.ImportError.builder()
                        .row(rowNum)
                        .message(e.getMessage())
                        .build());
                if (!partial) {
                    // 全量模式：任何失败都回滚
                    throw new RuntimeException("第 " + rowNum + " 行导入失败: " + e.getMessage()
                            + "（全量模式已回滚所有数据）", e);
                }
            }
        }

        return ImportResult.builder()
                .totalRows(total)
                .successCount(successCount)
                .failureCount(failureCount)
                .skippedCount(0)
                .errors(errors)
                .build();
    }

    private String validateRow(Map<String, Object> row) {
        String username = getAsString(row, "username");
        if (CharSequenceUtil.isBlank(username)) {
            return "用户名为空";
        }
        long exists = userService.count(new LambdaQueryWrapper<SysUser>().eq(SysUser::getUsername, username));
        if (exists > 0) {
            return "用户名已存在: " + username;
        }

        String nickname = getAsString(row, "nickname");
        if (CharSequenceUtil.isBlank(nickname)) {
            return "昵称为空";
        }

        String mobile = getAsString(row, "mobile");
        if (CharSequenceUtil.isBlank(mobile)) {
            return "手机号为空";
        }
        if (!Validator.isMobile(mobile)) {
            return "手机号格式不正确: " + mobile;
        }

        String email = getAsString(row, "email");
        if (CharSequenceUtil.isNotBlank(email) && !Validator.isEmail(email)) {
            return "邮箱格式不正确: " + email;
        }

        String genderLabel = getAsString(row, "genderLabel");
        if (CharSequenceUtil.isNotBlank(genderLabel)
                && IBaseEnum.getValueByLabel(genderLabel, GenderEnum.class) == null) {
            return "性别取值无效（应为 男/女）: " + genderLabel;
        }

        return null;
    }

    private SysUser buildEntity(Map<String, Object> row, Long deptId, String defaultPassword,
                                Map<String, Long> roleCodeMap) {
        SysUser entity = new SysUser();
        entity.setUsername(getAsString(row, "username"));
        entity.setNickname(getAsString(row, "nickname"));
        entity.setMobile(getAsString(row, "mobile"));
        entity.setEmail(getAsString(row, "email"));
        entity.setDeptId(deptId);
        entity.setPassword(defaultPassword);
        entity.setStatus(StatusEnum.ENABLE.getValue());

        String genderLabel = getAsString(row, "genderLabel");
        if (CharSequenceUtil.isNotBlank(genderLabel)) {
            Object genderValue = IBaseEnum.getValueByLabel(genderLabel, GenderEnum.class);
            if (genderValue instanceof Integer integer) {
                entity.setGender(integer);
            }
        }
        return entity;
    }

    private List<Long> parseRoleIds(Map<String, Object> row, Map<String, Long> roleCodeMap) {
        String roleCodes = getAsString(row, "roleCodes");
        if (CharSequenceUtil.isBlank(roleCodes)) {
            return List.of();
        }
        return Arrays.stream(roleCodes.split(","))
                .map(String::trim)
                .filter(CharSequenceUtil::isNotBlank)
                .map(roleCodeMap::get)
                .filter(java.util.Objects::nonNull)
                .toList();
    }

    private Map<String, Long> loadRoleCodeMap(List<Map<String, Object>> rows) {
        List<String> codes = rows.stream()
                .map(r -> getAsString(r, "roleCodes"))
                .filter(CharSequenceUtil::isNotBlank)
                .flatMap(s -> Arrays.stream(s.split(",")))
                .map(String::trim)
                .filter(CharSequenceUtil::isNotBlank)
                .distinct()
                .toList();
        if (codes.isEmpty()) {
            return Map.of();
        }
        return roleService.list(new LambdaQueryWrapper<SysRole>()
                        .in(SysRole::getCode, codes)
                        .eq(SysRole::getStatus, StatusEnum.ENABLE.getValue()))
                .stream()
                .collect(Collectors.toMap(SysRole::getCode, SysRole::getId, (a, b) -> a));
    }

    private String getAsString(Map<String, Object> row, String key) {
        Object v = row.get(key);
        return v == null ? null : String.valueOf(v).trim();
    }

    private Long extractLong(Map<String, Object> params, String key) {
        if (params == null) {
            return null;
        }
        Object v = params.get(key);
        if (v == null || "".equals(String.valueOf(v))) {
            return null;
        }
        return Long.valueOf(String.valueOf(v));
    }
}
