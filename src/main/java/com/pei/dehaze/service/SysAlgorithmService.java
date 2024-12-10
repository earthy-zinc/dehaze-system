package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:34:16
 */
public interface SysAlgorithmService extends IService<SysAlgorithm> {
    List<AlgorithmVO> getList(AlgorithmQuery queryParams);

    List<Option<Long>> getOption();

    SysAlgorithm getAlgorithmById(Long id);

    SysAlgorithm getRootAlgorithm(Long id);

    boolean addAlgorithm(AlgorithmForm algorithm);

    boolean updateAlgorithm(AlgorithmForm algorithm);

    boolean deleteAlgorithms(List<Long> ids);
}
