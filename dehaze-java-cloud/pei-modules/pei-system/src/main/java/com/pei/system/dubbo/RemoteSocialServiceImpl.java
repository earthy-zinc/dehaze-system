package com.pei.system.dubbo;

import com.pei.system.domain.bo.SysSocialBo;
import com.pei.system.domain.vo.SysSocialVo;
import com.pei.system.service.ISysSocialService;
import lombok.RequiredArgsConstructor;
import org.apache.dubbo.config.annotation.DubboService;
import com.pei.common.core.utils.MapstructUtils;
import com.pei.system.api.RemoteSocialService;
import com.pei.system.api.domain.bo.RemoteSocialBo;
import com.pei.system.api.domain.vo.RemoteSocialVo;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 社会化关系服务
 *
 * @author Michelle.Chung
 */
@RequiredArgsConstructor
@Service
@DubboService
public class RemoteSocialServiceImpl implements RemoteSocialService {

    private final ISysSocialService sysSocialService;

    /**
     * 根据 authId 查询用户授权信息
     *
     * @param authId 认证id
     * @return 授权信息
     */
    @Override
    public List<RemoteSocialVo> selectByAuthId(String authId) {
        List<SysSocialVo> list = sysSocialService.selectByAuthId(authId);
        return MapstructUtils.convert(list, RemoteSocialVo.class);
    }

    /**
     * 查询列表
     *
     * @param bo 社会化关系业务对象
     */
    @Override
    public List<RemoteSocialVo> queryList(RemoteSocialBo bo) {
        SysSocialBo params = MapstructUtils.convert(bo, SysSocialBo.class);
        List<SysSocialVo> list = sysSocialService.queryList(params);
        return MapstructUtils.convert(list, RemoteSocialVo.class);
    }

    /**
     * 保存社会化关系
     *
     * @param bo 社会化关系业务对象
     */
    @Override
    public void insertByBo(RemoteSocialBo bo) {
        sysSocialService.insertByBo(MapstructUtils.convert(bo, SysSocialBo.class));
    }

    /**
     * 更新社会化关系
     *
     * @param bo 社会化关系业务对象
     */
    @Override
    public void updateByBo(RemoteSocialBo bo) {
        sysSocialService.updateByBo(MapstructUtils.convert(bo, SysSocialBo.class));
    }

    /**
     * 删除社会化关系
     *
     * @param socialId 社会化关系ID
     * @return 结果
     */
    @Override
    public Boolean deleteWithValidById(Long socialId) {
        return sysSocialService.deleteWithValidById(socialId);
    }

}
