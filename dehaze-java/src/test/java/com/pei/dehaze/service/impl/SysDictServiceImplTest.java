package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.converter.DictConverter;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.mapper.SysDictTypeMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.RedisTemplate;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;

/**
 * SysDictServiceImpl#getIntValue 单元测试。
 * <p>
 * 复用 spy 对 listDictOptions 打桩，验证字典整型值读取的映射与缺键/非法值回退逻辑。
 */
@DisplayName("SysDictServiceImpl#getIntValue 单元测试")
@ExtendWith(MockitoExtension.class)
class SysDictServiceImplTest {

    private SysDictServiceImpl buildSpy(List<Option<String>> options) {
        SysDictServiceImpl service = new SysDictServiceImpl(
                mock(DictConverter.class),
                mock(SysDictTypeMapper.class),
                mock(RedisTemplate.class));
        SysDictServiceImpl spy = spy(service);
        lenient().doReturn(options).when(spy).listDictOptions("favorite_capacity");
        lenient().doReturn(options).when(spy).listDictOptions("member_growth_rules");
        return spy;
    }

    private List<Option<String>> capacityOptions() {
        return List.of(
                new Option<>("200", "default"),
                new Option<>("500", "vip1"),
                new Option<>("1000", "vip2"),
                new Option<>("3000", "svip"));
    }

    @Test
    @DisplayName("getIntValue - 按键(name)读取对应的整型值(value)")
    void getIntValue_readsValueByKey() {
        SysDictServiceImpl service = buildSpy(capacityOptions());
        assertThat(service.getIntValue("favorite_capacity", "default", 200)).isEqualTo(200);
        assertThat(service.getIntValue("favorite_capacity", "vip1", 200)).isEqualTo(500);
        assertThat(service.getIntValue("favorite_capacity", "vip2", 200)).isEqualTo(1000);
        assertThat(service.getIntValue("favorite_capacity", "svip", 200)).isEqualTo(3000);
    }

    @Test
    @DisplayName("getIntValue - 缺键时回退默认值")
    void getIntValue_missingKeyFallsBack() {
        SysDictServiceImpl service = buildSpy(capacityOptions());
        assertThat(service.getIntValue("favorite_capacity", "vip3", 200)).isEqualTo(200);
    }

    @Test
    @DisplayName("getIntValue - 无字典项时回退默认值")
    void getIntValue_emptyOptionsFallsBack() {
        SysDictServiceImpl service = buildSpy(List.of());
        assertThat(service.getIntValue("member_growth_rules", "sign_in_value", 3)).isEqualTo(3);
    }

    @Test
    @DisplayName("getIntValue - 数值非法(非数字)时回退默认值")
    void getIntValue_nonNumericFallsBack() {
        SysDictServiceImpl service = buildSpy(List.of(new Option<>("abc", "sign_in_value")));
        assertThat(service.getIntValue("member_growth_rules", "sign_in_value", 3)).isEqualTo(3);
    }
}
