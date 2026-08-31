package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysFavoriteMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.service.SysDictService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

/**
 * FavoriteServiceImpl 收藏容量（sys_dict: favorite_capacity）单元测试。
 * <p>
 * 验证会员等级 → 字典键（level_0→default / level_1→vip1 / level_2→vip2 / level_3→svip）的映射。
 */
@DisplayName("FavoriteServiceImpl 收藏容量字典化单元测试")
@ExtendWith(MockitoExtension.class)
class FavoriteServiceImplTest {

    @Mock
    private SysMemberMapper memberMapper;
    @Mock
    private SysAlgorithmMapper sysAlgorithmMapper;
    @Mock
    private SysDatasetMapper sysDatasetMapper;
    @Mock
    private SysPredLogMapper sysPredLogMapper;
    @Mock
    private SysDictService sysDictService;

    private FavoriteServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new FavoriteServiceImpl(memberMapper, sysAlgorithmMapper,
                sysDatasetMapper, sysPredLogMapper, sysDictService);
        // 按字典键返回对应容量，用于验证等级→键映射
        lenient().when(sysDictService.getIntValue(eq("favorite_capacity"), eq("default"), any(Integer.class))).thenReturn(200);
        lenient().when(sysDictService.getIntValue(eq("favorite_capacity"), eq("vip1"), any(Integer.class))).thenReturn(500);
        lenient().when(sysDictService.getIntValue(eq("favorite_capacity"), eq("vip2"), any(Integer.class))).thenReturn(1000);
        lenient().when(sysDictService.getIntValue(eq("favorite_capacity"), eq("svip"), any(Integer.class))).thenReturn(3000);
    }

    private void stubMember(Long userId, String levelCode) {
        SysMember member = new SysMember();
        member.setUserId(userId);
        member.setLevelCode(levelCode);
        when(memberMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(member);
    }

    @Test
    @DisplayName("getCapacity - 普通用户(level_0)映射 default 容量 200")
    void level0_mapsDefault() {
        stubMember(1L, "level_0");
        assertThat(service.getCapacity(1L)).isEqualTo(200);
    }

    @Test
    @DisplayName("getCapacity - VIP1(level_1)映射 vip1 容量 500")
    void level1_mapsVip1() {
        stubMember(1L, "level_1");
        assertThat(service.getCapacity(1L)).isEqualTo(500);
    }

    @Test
    @DisplayName("getCapacity - VIP2(level_2)映射 vip2 容量 1000")
    void level2_mapsVip2() {
        stubMember(1L, "level_2");
        assertThat(service.getCapacity(1L)).isEqualTo(1000);
    }

    @Test
    @DisplayName("getCapacity - SVIP(level_3)映射 svip 容量 3000")
    void level3_mapsSvip() {
        stubMember(1L, "level_3");
        assertThat(service.getCapacity(1L)).isEqualTo(3000);
    }

    @Test
    @DisplayName("getCapacity - 无会员记录时按普通用户(default)容量")
    void noMember_mapsDefault() {
        when(memberMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(null);
        assertThat(service.getCapacity(1L)).isEqualTo(200);
    }

    @Test
    @DisplayName("getCapacity - 字典缺键时回退默认值 200")
    void missingDictKey_fallsBackToDefault() {
        stubMember(1L, "level_0");
        // 覆盖 default 键的桩为缺失(返回默认值)
        when(sysDictService.getIntValue(eq("favorite_capacity"), eq("default"), any(Integer.class))).thenReturn(200);
        assertThat(service.getCapacity(1L)).isEqualTo(200);
    }
}
