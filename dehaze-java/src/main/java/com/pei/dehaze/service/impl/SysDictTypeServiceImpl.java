package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.converter.DictTypeConverter;
import com.pei.dehaze.mapper.SysDictTypeMapper;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.model.entity.SysDictType;
import com.pei.dehaze.model.form.DictTypeForm;
import com.pei.dehaze.model.query.DictTypePageQuery;
import com.pei.dehaze.model.vo.DictTypePageVO;
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
                                SysDictType::getRemark
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
        Assert.isTrue(entity != null, "字典类型不存在");

        // 实体转换
        return dictTypeConverter.entity2Form(entity);
    }

    /**
     * 新增字典类型
     */
    @Override
    public boolean saveDictType(DictTypeForm dictTypeForm) {
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
    public boolean updateDictType(Long id, DictTypeForm dictTypeForm) {
        // 获取字典类型
        SysDictType sysDictType = this.getById(id);
        Assert.isTrue(sysDictType != null, "字典类型不存在");

        SysDictType entity = dictTypeConverter.form2Entity(dictTypeForm);
        boolean result = this.updateById(entity);
        if (sysDictType != null && result) {
            // 字典类型code变化，同步修改字典项的类型code
            String oldCode = sysDictType.getCode();
            String newCode = dictTypeForm.getCode();
            if (!CharSequenceUtil.equals(oldCode, newCode)) {
                dictItemService.update(new LambdaUpdateWrapper<SysDict>()
                        .eq(SysDict::getTypeCode, oldCode)
                        .set(SysDict::getTypeCode, newCode)
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

        Assert.isTrue(CharSequenceUtil.isNotBlank(idsStr), "删除数据为空");

        List<String> ids = Arrays.stream(idsStr.split(",")).toList();

        // 删除字典数据项
        List<String> dictTypeCodes = this.list(new LambdaQueryWrapper<SysDictType>()
                        .in(SysDictType::getId, ids)
                        .select(SysDictType::getCode))
                .stream()
                .map(SysDictType::getCode)
                .toList();
        if (CollUtil.isNotEmpty(dictTypeCodes)) {
            dictItemService.remove(new LambdaQueryWrapper<SysDict>()
                    .in(SysDict::getTypeCode, dictTypeCodes));
        }
        // 删除字典类型
        return this.removeByIds(ids);
    }

    /**
     * 获取字典类型的数据项
     */
    @Override
    public List<Option<String>> listDictItemsByTypeCode(String typeCode) {
        // 数据字典项
        List<SysDict> dictItems = dictItemService.list(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, typeCode)
                .select(SysDict::getValue, SysDict::getName)
        );

        // 转换下拉数据
        return CollUtil.emptyIfNull(dictItems)
                .stream()
                .map(dictItem -> new Option<>(dictItem.getValue(), dictItem.getName()))
                .toList();
    }


}




