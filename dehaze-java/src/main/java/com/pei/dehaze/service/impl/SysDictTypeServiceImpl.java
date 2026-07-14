package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
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
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Arrays;
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


    private final SysDictService dictItemService;
    private final DictTypeConverter dictTypeConverter;

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
                        .like(CharSequenceUtil.isNotBlank(keywords), SysDictType::getName, keywords)
                        .or()
                        .like(CharSequenceUtil.isNotBlank(keywords), SysDictType::getCode, keywords)
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
        return dictTypeConverter.entity2Form(entity);
    }

    /**
     * 新增字典类型
     */
    @Override
    public boolean saveDictType(DictTypeForm dictTypeForm) {
        // 检查编码唯一性
        long count = this.count(new LambdaQueryWrapper<SysDictType>()
                .eq(SysDictType::getCode, dictTypeForm.getCode()));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "字典类型编码已存在");
        }
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

        // 检查编码唯一性（排除自身ID）
        long count = this.count(new LambdaQueryWrapper<SysDictType>()
                .eq(SysDictType::getCode, dictTypeForm.getCode())
                .ne(SysDictType::getId, id));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS);
        }

        SysDictType entity = dictTypeConverter.form2Entity(dictTypeForm);
        entity.setId(id);  // 设置ID，确保更新正确执行
        boolean result = this.updateById(entity);
        if (result) {
            // 字典类型code变化，同步修改字典项的类型code
            String oldCode = sysDictType.getCode();
            String newCode = dictTypeForm.getCode();
            if (!CharSequenceUtil.equals(oldCode, newCode)) {
                Long currentUserId = SecurityUtils.getUserId();
                dictItemService.update(new LambdaUpdateWrapper<SysDict>()
                        .eq(SysDict::getTypeCode, oldCode)
                        .set(SysDict::getTypeCode, newCode)
                        .set(SysDict::getUpdateBy, currentUserId)
                );
            }
        }
        return result;
    }

    /**
     * 删除字典类型
     *
     * @param idsStr 字典类型ID，多个以英文逗号(,)分割
     */
    @Override
    @Transactional
    public boolean deleteDictTypes(String idsStr) {

        if (CharSequenceUtil.isBlank(idsStr)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "删除数据为空");
        }

        // 转换ID列表，校验非数字ID
        List<Long> ids;
        try {
            ids = Arrays.stream(idsStr.split(","))
                    .map(Long::parseLong)
                    .toList();
        } catch (NumberFormatException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "ID格式错误");
        }

        // 校验字典类型是否存在
        long existCount = this.count(new LambdaQueryWrapper<SysDictType>()
                .in(SysDictType::getId, ids));
        if (existCount == 0) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在");
        }

        // 获取字典类型编码列表
        List<String> dictTypeCodes = this.list(new LambdaQueryWrapper<SysDictType>()
                        .in(SysDictType::getId, ids)
                        .select(SysDictType::getCode))
                .stream()
                .map(SysDictType::getCode)
                .toList();

        // 校验字典类型下是否有字典数据
        if (CollUtil.isNotEmpty(dictTypeCodes)) {
            long dictCount = dictItemService.count(new LambdaQueryWrapper<SysDict>()
                    .in(SysDict::getTypeCode, dictTypeCodes));
            if (dictCount > 0) {
                throw new BusinessException(ResultCode.DATA_BIND_EXISTS, "存在关联的字典数据，无法删除");
            }
        }
        // 删除字典类型
        return this.removeByIds(ids);
    }

}




