package com.pei.dehaze.base;

import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.transaction.annotation.Transactional;

/**
 * 测试基类
 * 
 * 提供基础的 Spring Boot 测试配置
 * - 使用 test profile
 * - 自动回滚事务（保证测试隔离性）
 * - H2 内存数据库（快速、无状态）
 * 
 * @author earthyzinc
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest
@ActiveProfiles("test")
@Transactional
public abstract class BaseTest {

}
