package com.pei.dehaze.common;

import lombok.ToString;

import java.util.ArrayList;
import java.util.List;

public class ListTest {

    public static void main(String[] args) {
        // 示例初始化a和b列表
        List<Integer> a = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6, 7, 8)); // 假设n=4时
        List<Integer> b = new ArrayList<>(List.of(9, 10));

        int x = b.size(); // b的长度
        int n = a.size() / x; // 计算n

        ArrayList<PairImage> pairImages = new ArrayList<>();
        for (int i = 0; i < x; i++) {
            List<Integer> haze = new ArrayList<>();
            Integer clean = b.remove(0);
            for (int j = 0; j < n && !a.isEmpty(); j++) {
                haze.add(a.remove(0));
            }
            PairImage pairImage = new PairImage();
            pairImage.cleanPath = clean;
            pairImage.hazePath = haze;
            pairImages.add(pairImage);
        }
        System.out.println(pairImages);
    }

    @ToString
    public static class PairImage {
        List<Integer> hazePath;
        Integer cleanPath;
    }
}
