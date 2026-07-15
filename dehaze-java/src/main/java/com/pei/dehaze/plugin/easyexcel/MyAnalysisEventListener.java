package com.pei.dehaze.plugin.easyexcel;

import com.alibaba.excel.event.AnalysisEventListener;

/**
 * 自定义解析结果监听器
 *
 * @author earthyzinc
 * @since 2023/03/01
 */
public abstract class MyAnalysisEventListener<T> extends AnalysisEventListener<T> {

    public abstract String getMsg();
}
