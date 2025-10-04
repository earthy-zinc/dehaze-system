package com.pei.dehaze.module.member.api.address;

import com.pei.dehaze.framework.common.pojo.CommonResult;
import com.pei.dehaze.module.member.api.address.dto.MemberAddressRespDTO;
import com.pei.dehaze.module.member.convert.address.AddressConvert;
import com.pei.dehaze.module.member.service.address.AddressService;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.Resource;

import static com.pei.dehaze.framework.common.pojo.CommonResult.success;

/**
 * 用户收件地址 API 实现类
 *
 * @author earthyzinc
 */
@RestController // 提供 RESTful API 接口，给 Feign 调用
@Validated
public class MemberAddressApiImpl implements MemberAddressApi {

    @Resource
    private AddressService addressService;

    @Override
    public CommonResult<MemberAddressRespDTO> getAddress(Long id, Long userId) {
        return success(AddressConvert.INSTANCE.convert02(addressService.getAddress(userId, id)));
    }

    @Override
    public CommonResult<MemberAddressRespDTO> getDefaultAddress(Long userId) {
        return success(AddressConvert.INSTANCE.convert02(addressService.getDefaultUserAddress(userId)));
    }

}
