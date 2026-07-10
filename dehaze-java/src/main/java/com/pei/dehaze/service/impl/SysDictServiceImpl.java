package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.converter.DictConverter;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.mapper.SysDictTypeMapper;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.model.entity.SysDictType;
import com.pei.dehaze.model.form.DictForm;
import com.pei.dehaze.model.query.DictPageQuery;
import com.pei.dehaze.model.vo.DictPageVO;
import com.pei.dehaze.service.SysDictService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

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
    private final SysDictTypeMapper dictTypeMapper;
    private final RedisTemplate<String, Object> redisTemplate;

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

        // typeCode 必填校验
        if (CharSequenceUtil.isBlank(typeCode)) {
            throw new BusinessException(ResultCode.PARAM_IS_NULL);
        }

        // 查询数据
        Page<SysDict> dictItemPage = this.page(
                new Page<>(pageNum, pageSize),
                new LambdaQueryWrapper<SysDict>()
                        .like(CharSequenceUtil.isNotBlank(keywords), SysDict::getName, keywords)
                        .eq(SysDict::getTypeCode, typeCode)
                        .select(SysDict::getId, SysDict::getName, SysDict::getValue,
                                SysDict::getTypeCode, SysDict::getDefaulted, SysDict::getSort,
                                SysDict::getStatus, SysDict::getRemark, SysDict::getCreateTime)
                        .orderByAsc(SysDict::getSort)
                        .orderByDesc(SysDict::getCreateTime)
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
                        SysDict::getDefaulted,
                        SysDict::getRemark
                ));
        if (entity == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

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
        String typeCode = dictForm.getTypeCode();

        // 类型存在性检查
        long typeCount = dictTypeMapper.selectCount(new LambdaQueryWrapper<SysDictType>()
                .eq(SysDictType::getCode, typeCode));
        if (typeCount == 0) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // 唯一性检查（同类型下 value 唯一）
        long count = this.count(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, typeCode)
                .eq(SysDict::getValue, dictForm.getValue()));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS);
        }

        // 实体对象转换 form->entity
        SysDict entity = dictConverter.form2Entity(dictForm);
        // 持久化
        boolean result = this.save(entity);

        // 清除缓存
        redisTemplate.delete("dict:options:" + typeCode);

        return result;
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
        // 获取字典数据项
        SysDict existDict = this.getById(id);
        if (existDict == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // typeCode 只读，保留原记录的 typeCode
        dictForm.setTypeCode(existDict.getTypeCode());

        // 唯一性检查（同类型下 value 唯一，排除自身）
        long count = this.count(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, existDict.getTypeCode())
                .eq(SysDict::getValue, dictForm.getValue())
                .ne(SysDict::getId, id));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS);
        }

        // 实体对象转换 form->entity
        SysDict entity = dictConverter.form2Entity(dictForm);
        entity.setId(id);  // 设置ID，确保更新正确执行
        boolean result = this.updateById(entity);

        // 清除缓存
        redisTemplate.delete("dict:options:" + existDict.getTypeCode());

        return result;
    }

    /**
     * 删除字典数据项
     *
     * @param idsStr 字典数据项ID，多个以英文逗号(,)分割
     * @return
     */
    @Override
    public boolean deleteDict(String idsStr) {
        if (CharSequenceUtil.isBlank(idsStr)) {
            throw new BusinessException(ResultCode.PARAM_ERROR);
        }
        List<Long> ids = Arrays.stream(idsStr.split(","))
                .map(Long::parseLong)
                .toList();

        // 校验字典数据项是否存在
        long existCount = this.count(new LambdaQueryWrapper<SysDict>()
                .in(SysDict::getId, ids));
        if (existCount == 0) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // 查出受影响的 typeCode（用于缓存清理）
        List<SysDict> dicts = this.list(new LambdaQueryWrapper<SysDict>()
                .in(SysDict::getId, ids)
                .select(SysDict::getTypeCode));

        // 删除字典数据项
        boolean result = this.removeByIds(ids);

        // 清除缓存
        for (SysDict dict : dicts) {
            if (CharSequenceUtil.isNotBlank(dict.getTypeCode())) {
                redisTemplate.delete("dict:options:" + dict.getTypeCode());
            }
        }

        return result;
    }

    /**
     * 获取字典下拉列表
     *
     * @param typeCode
     * @return
     */
    @Override
    public List<Option<String>> listDictOptions(String typeCode) {
        String cacheKey = "dict:options:" + typeCode;

        // 查缓存
        Object cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached instanceof List) {
            @SuppressWarnings("unchecked")
            List<Option<String>> cachedOptions = (List<Option<String>>) cached;
            return cachedOptions;
        }

        // 查询数据字典项（只返回启用状态，按 sort 和 create_time 排序）
        List<SysDict> dictList = this.list(new LambdaQueryWrapper<SysDict>()
                .eq(SysDict::getTypeCode, typeCode)
                .eq(SysDict::getStatus, 1)
                .select(SysDict::getValue, SysDict::getName)
                .orderByAsc(SysDict::getSort)
                .orderByDesc(SysDict::getCreateTime));

        // 转换下拉数据
        List<Option<String>> options = CollUtil.emptyIfNull(dictList)
                .stream()
                .map(dictItem -> new Option<>(dictItem.getValue(), dictItem.getName()))
                .toList();

        // 写缓存（非空结果才缓存）
        if (!options.isEmpty()) {
            redisTemplate.opsForValue().set(cacheKey, options, 1, TimeUnit.HOURS);
        }

        return options;
    }
}




