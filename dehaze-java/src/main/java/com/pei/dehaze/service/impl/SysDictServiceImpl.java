package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.converter.DictConverter;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.model.form.DictForm;
import com.pei.dehaze.model.query.DictPageQuery;
import com.pei.dehaze.model.vo.DictPageVO;
import com.pei.dehaze.service.SysDictService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

/**
 * 数据字典项业务实现类
 *
 * @author earthyzinc
 * @since 2022/10/12
 */
@Service
@RequiredArgsConstructor
public class SysDictServiceImpl extends ServiceImpl<SysDictMapper, SysDict> implements SysDictService {

    private final DictConverter dictConverter;

    /**
     * 字典数据项分页列表
     *
     * @param queryParams
     * @return
     */
    @Override
    public Page<DictPageVO> getDictPage(DictPageQuery queryParams) {
        // 查询参数
        int pageNum = queryParams.getPageNum();
        int pageSize = queryParams.getPageSize();
        String keywords = queryParams.getKeywords();
        String typeCode = queryParams.getTypeCode();

        // 查询数据
        Page<SysDict> dictItemPage = this.page(
                new Page<>(pageNum, pageSize),
                new LambdaQueryWrapper<SysDict>()
                        .like(CharSequenceUtil.isNotBlank(keywords), SysDict::getName, keywords)
                        .eq(CharSequenceUtil.isNotBlank(typeCode), SysDict::getTypeCode, typeCode)
                        .select(SysDict::getId, SysDict::getName, SysDict::getValue, SysDict::getStatus)
        );

        // 实体转换
        return dictConverter.entity2Page(dictItemPage);
    }

    /**
     * 字典数据项表单详情
     *
     * @param id 字典数据项ID
     * @return
     */
    @Override
    public DictForm getDictForm(Long id) {
        // 获取entity
        SysDict entity = this.getOne(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getId, id)
                .select(
                        SysDict::getId,
                        SysDict::getTypeCode,
                        SysDict::getName,
                        SysDict::getValue,
                        SysDict::getStatus,
                        SysDict::getSort,
                        SysDict::getRemark
                ));
        Assert.isTrue(entity != null, "字典数据项不存在");

        // 实体转换
        return dictConverter.entity2Form(entity);
    }

    /**
     * 新增字典数据项
     *
     * @param dictForm 字典数据项表单
     * @return
     */
    @Override
    public boolean saveDict(DictForm dictForm) {
        // 实体对象转换 form->entity
        SysDict entity = dictConverter.form2Entity(dictForm);
        // 持久化
        return this.save(entity);
    }

    /**
     * 修改字典数据项
     *
     * @param id           字典数据项ID
     * @param dictForm 字典数据项表单
     * @return
     */
    @Override
    public boolean updateDict(Long id, DictForm dictForm) {
        SysDict entity = dictConverter.form2Entity(dictForm);
        return this.updateById(entity);
    }

    /**
     * 删除字典数据项
     *
     * @param idsStr 字典数据项ID，多个以英文逗号(,)分割
     * @return
     */
    @Override
    public boolean deleteDict(String idsStr) {
        Assert.isTrue(CharSequenceUtil.isNotBlank(idsStr), "删除数据为空");
        //
        List<Long> ids = Arrays.stream(idsStr.split(","))
                .map(Long::parseLong)
                .toList();

        // 删除字典数据项
        return this.removeByIds(ids);
    }

    /**
     * 获取字典下拉列表
     *
     * @param typeCode
     * @return
     */
    @Override
    public List<Option<String>> listDictOptions(String typeCode) {
        // 数据字典项
        List<SysDict> dictList = this.list(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, typeCode)
                .select(SysDict::getValue, SysDict::getName)
        );

        // 转换下拉数据
        return CollUtil.emptyIfNull(dictList)
                .stream()
                .map(dictItem -> new Option<>(dictItem.getValue(), dictItem.getName()))
                .toList();
    }
}




