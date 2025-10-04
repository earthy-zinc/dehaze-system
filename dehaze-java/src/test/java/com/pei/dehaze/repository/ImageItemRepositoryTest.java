package com.pei.dehaze.repository;

import com.pei.dehaze.model.vo.ImageItemVO;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ImageItemRepositoryTest {
    @Mock
    private ImageItemRepository imageItemRepository;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    void testFindByDatasetId() {
        // 准备测试数据
        ImageItemVO item1 = new ImageItemVO(); // 假设默认构造函数初始化了必要的字段
        ImageItemVO item2 = new ImageItemVO();

        List<ImageItemVO> expectedItems = Arrays.asList(item1, item2);

        // 模拟Pageable对象，实际测试时可根据需要定制
        Pageable pageable = Pageable.ofSize(10).withPage(0);

        // 构造预期的Page对象
        Page<ImageItemVO> expectedPage = new PageImpl<>(expectedItems, pageable, expectedItems.size());

        // 配置mock行为
        when(imageItemRepository.findByDatasetId(1L, pageable)).thenReturn(expectedPage);

        // 调用待测试的方法
        Page<ImageItemVO> resultPage = imageItemRepository.findByDatasetId(1L, pageable);

        // 验证结果
        assertEquals(expectedPage, resultPage);
        assertEquals(expectedItems.size(), resultPage.getNumberOfElements());
    }
}
