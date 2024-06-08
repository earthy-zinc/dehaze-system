package com.pei.dehaze.common.util;

import com.alibaba.excel.EasyExcel;
import com.pei.dehaze.plugin.easyexcel.MyAnalysisEventListener;

import java.io.InputStream;

/**
 * Excel 工具类
 *
 * @author earthyzinc
 * @since 2023/03/01
 */
public class ExcelUtils {

    public static <T> String importExcel(InputStream is, Class clazz, MyAnalysisEventListener<T> listener) {
        EasyExcel.read(is, clazz, listener).sheet().doRead();
        return listener.getMsg();
    }
}
