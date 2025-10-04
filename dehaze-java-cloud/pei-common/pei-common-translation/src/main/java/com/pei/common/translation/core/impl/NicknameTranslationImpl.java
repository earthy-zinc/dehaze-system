package com.pei.common.translation.core.impl;

import com.pei.common.translation.annotation.TranslationType;
import com.pei.common.translation.constant.TransConstant;
import com.pei.common.translation.core.TranslationInterface;
import lombok.AllArgsConstructor;
import org.apache.dubbo.config.annotation.DubboReference;
import com.pei.system.api.RemoteUserService;

/**
 * 用户昵称翻译实现
 *
 * @author may
 */
@AllArgsConstructor
@TranslationType(type = TransConstant.USER_ID_TO_NICKNAME)
public class NicknameTranslationImpl implements TranslationInterface<String> {

    @DubboReference
    private RemoteUserService remoteUserService;

    @Override
    public String translation(Object key, String other) {
        if (key instanceof Long id) {
            return remoteUserService.selectNicknameByIds(id.toString());
        } else if (key instanceof String ids) {
            return remoteUserService.selectNicknameByIds(ids);
        }
        return null;
    }
}
