package com.pei.system.dubbo;

import com.pei.system.domain.vo.SysDictDataVo;
import com.pei.system.domain.vo.SysDictTypeVo;
import com.pei.system.service.ISysDictTypeService;
import lombok.RequiredArgsConstructor;
import org.apache.dubbo.config.annotation.DubboService;
import com.pei.common.core.utils.MapstructUtils;
import com.pei.system.api.RemoteDictService;
import com.pei.system.api.domain.vo.RemoteDictDataVo;
import com.pei.system.api.domain.vo.RemoteDictTypeVo;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 字典服务
 *
 * @author Lion Li
 */
@RequiredArgsConstructor
@Service
@DubboService
public class RemoteDictServiceImpl implements RemoteDictService {

    private final ISysDictTypeService sysDictTypeService;

    @Override
    public RemoteDictTypeVo selectDictTypeByType(String dictType) {
        SysDictTypeVo vo = sysDictTypeService.selectDictTypeByType(dictType);
        return MapstructUtils.convert(vo, RemoteDictTypeVo.class);
    }

    /**
     * 根据字典类型查询字典数据
     *
     * @param dictType 字典类型
     * @return 字典数据集合信息
     */
    @Override
    public List<RemoteDictDataVo> selectDictDataByType(String dictType) {
        List<SysDictDataVo> list = sysDictTypeService.selectDictDataByType(dictType);
        return MapstructUtils.convert(list, RemoteDictDataVo.class);
    }

}
