package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.IdUtils;
import com.pei.dehaze.converter.DictTypeConverter;
import com.pei.dehaze.mapper.SysDictTypeMapper;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.model.entity.SysDictType;
import com.pei.dehaze.model.form.DictTypeForm;
import com.pei.dehaze.model.query.DictTypePageQuery;
import com.pei.dehaze.model.vo.DictTypePageVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysDictService;
import com.pei.dehaze.service.SysDictTypeService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 数据字典类型业务实现类
 *
 * @author earthyzinc
 * @since 2022/10/12
 */
@Service
@RequiredArgsConstructor
public class SysDictTypeServiceImpl extends ServiceImpl<SysDictTypeMapper, SysDictType> implements SysDictTypeService {

    /** 系统预置字典类型编码（种子内置，对齐 python SYSTEM_PRESET_DICT_TYPE_CODES），不可删除 */
    private static final List<String> SYSTEM_PRESET_DICT_TYPE_CODES = List.of(
            "gender", "ai_guardrail_defaults", "ai_provider_health", "ai_embedding",
            "member_growth_rules", "favorite_capacity", "ai_eval");

    private static final String DICT_OPTIONS_CACHE_KEY_PREFIX = "dict:data:";


    private final SysDictService dictItemService;
    private final DictTypeConverter dictTypeConverter;
    private final SysDictTypeMapper dictTypeMapper;
    private final StringRedisTemplate stringRedisTemplate;

    /**
     * 字典分页列表
     *
     * @param queryParams 分页查询对象
     */
    @Override
    public Page<DictTypePageVO> getDictTypePage(DictTypePageQuery queryParams) {
        // 查询参数
        int pageNum = queryParams.getPageNum();
        int pageSize = queryParams.getPageSize();
        String keywords = queryParams.getKeywords();

        // 查询数据
        Page<SysDictType> dictTypePage = this.page(
                new Page<>(pageNum, pageSize),
                new LambdaQueryWrapper<SysDictType>()
                        .and(CharSequenceUtil.isNotBlank(keywords),
                                wrapper -> wrapper
                                        .like(SysDictType::getName, keywords)
                                        .or()
                                        .like(SysDictType::getCode, keywords))
                        .select(
                                SysDictType::getId,
                                SysDictType::getName,
                                SysDictType::getCode,
                                SysDictType::getStatus,
                                SysDictType::getRemark,
                                SysDictType::getCreateTime
                        )
        );

        // 实体转换
        return dictTypeConverter.entity2Page(dictTypePage);
    }

    /**
     * 获取字典类型表单详情
     *
     * @param id 字典类型ID
     */
    @Override
    public DictTypeForm getDictTypeForm(Long id) {
        // 获取entity
        SysDictType entity = this.getOne(new LambdaQueryWrapper<SysDictType>()
                .eq(SysDictType::getId, id)
                .select(
                        SysDictType::getId,
                        SysDictType::getName,
                        SysDictType::getCode,
                        SysDictType::getStatus,
                        SysDictType::getRemark
                ));
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // 实体转换
        DictTypeForm form = dictTypeConverter.entity2Form(entity);
        // isPreset：系统预置类型标记（python 契约：预置 true / 新建 false）
        form.setIsPreset(SYSTEM_PRESET_DICT_TYPE_CODES.contains(entity.getCode()));
        return form;
    }

    /**
     * 新增字典类型
     */
    @Override
    public boolean saveDictType(DictTypeForm dictTypeForm) {
        // 检查编码唯一性
        validateCodeUnique(dictTypeForm.getCode(), null);
        // 实体对象转换 form->entity
        SysDictType entity = dictTypeConverter.form2Entity(dictTypeForm);
        // 持久化
        return this.save(entity);
    }


    /**
     * 修改字典类型
     *
     * @param id           字典类型ID
     * @param dictTypeForm 字典类型表单
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean updateDictType(Long id, DictTypeForm dictTypeForm) {
        // 获取字典类型
        SysDictType sysDictType = this.getById(id);
        if (sysDictType == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // code 只读：修改编码直接拒绝（T-DM-015，python A0503 口径）
        if (CharSequenceUtil.isNotBlank(dictTypeForm.getCode())
                && !CharSequenceUtil.equals(dictTypeForm.getCode(), sysDictType.getCode())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "字典类型编码不可修改");
        }

        SysDictType entity = dictTypeConverter.form2Entity(dictTypeForm);
        entity.setId(id);
        entity.setCode(sysDictType.getCode());
        boolean result = this.updateById(entity);
        if (result) {
            // 类型禁用/启用影响下拉可见性，失效下拉缓存
            stringRedisTemplate.delete(DICT_OPTIONS_CACHE_KEY_PREFIX + sysDictType.getCode());
        }
        return result;
    }

    /**
     * 校验字典类型编码唯一性（仅活跃行；唯一键含 deleted，软删行不占键位）。
     *
     * @param code      字典类型编码
     * @param excludeId 排除的字典类型ID（更新时传自身ID，新增时传 null）
     */
    private void validateCodeUnique(String code, Long excludeId) {
        long count = dictTypeMapper.selectCount(new LambdaQueryWrapper<SysDictType>()
                .eq(SysDictType::getCode, code)
                .ne(excludeId != null, SysDictType::getId, excludeId));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "字典类型编码已存在");
        }
    }

    /**
     * 删除字典类型
     *
     * @param idsStr 字典类型ID，多个以英文逗号(,)分割
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean deleteDictTypes(String idsStr, boolean force) {

        if (CharSequenceUtil.isBlank(idsStr)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "删除数据为空");
        }

        // 转换ID列表，校验非数字ID
        List<Long> ids = IdUtils.parseIdList(idsStr);

        // 校验字典类型是否存在
        long existCount = this.count(new LambdaQueryWrapper<SysDictType>()
                .in(SysDictType::getId, ids));
        if (existCount == 0) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在");
        }

        // 获取字典类型编码列表
        List<SysDictType> dictTypes = this.list(new LambdaQueryWrapper<SysDictType>()
                .in(SysDictType::getId, ids)
                .select(SysDictType::getCode));
        List<String> dictTypeCodes = dictTypes.stream().map(SysDictType::getCode).toList();

        // T-DM-025：系统预置字典类型不可删除
        if (dictTypes.stream().map(SysDictType::getCode).anyMatch(SYSTEM_PRESET_DICT_TYPE_CODES::contains)) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "系统预置字典类型不可删除");
        }

        if (CollUtil.isNotEmpty(dictTypeCodes)) {
            if (force) {
                dictItemService.deleteByTypeCodes(dictTypeCodes);
            } else {
                long dictCount = dictItemService.count(new LambdaQueryWrapper<SysDict>()
                        .in(SysDict::getTypeCode, dictTypeCodes));
                if (dictCount > 0) {
                    throw new BusinessException(ResultCode.DATA_BIND_EXISTS, "存在关联的字典数据，无法删除");
                }
            }
        }
        // 删除字典类型并失效相关下拉缓存
        boolean result = this.removeByIds(ids);
        if (result) {
            for (String code : dictTypeCodes) {
                if (CharSequenceUtil.isNotBlank(code)) {
                    stringRedisTemplate.delete(DICT_OPTIONS_CACHE_KEY_PREFIX + code);
                }
            }
        }
        return result;
    }

}




