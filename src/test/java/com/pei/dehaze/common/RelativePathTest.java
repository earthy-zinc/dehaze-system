package com.pei.dehaze.common;

import cn.hutool.core.io.file.PathUtil;
import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class RelativePathTest {

    @Test
    public void shouldReturnRelativePath() {
        // 定义基础路径和目标路径
        String baseDirectory = "D:\\DeepLearning\\dataset";
        String targetFile = "D:\\DeepLearning\\dataset\\RSHAZE\\train";

        // 将字符串路径转换为Path对象
        Path baseDirPath = Paths.get(baseDirectory);
        Path targetFilePath = Paths.get(targetFile);

        // 判断目标路径是否为基础路径的子路径
        assertTrue(targetFilePath.startsWith(baseDirPath));

        // 使用Hutool的isSub方法判断目标路径是否为基础路径的子路径
        assertTrue(PathUtil.isSub(baseDirPath, targetFilePath));

        // 计算目标路径相对于基础路径的相对路径
        Path relativePath = baseDirPath.relativize(targetFilePath);

        // 验证计算出的相对路径是否正确
        assertEquals("RSHAZE\\train", relativePath.toString());

        // 打印相对路径
        System.out.println("Relative path is: " + relativePath);
    }
}
