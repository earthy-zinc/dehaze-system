package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * 图像任务异步落库口径单元测试。
 * <p>
 * 结果图与原图的 MD5 都是**图片内容** MD5，由算法服务（Python）下载图片后算出并随响应返回；
 * Java 侧不得用 URL 的 MD5 冒充，也不得漏写 origin_md5。
 */
@DisplayName("PredLogAsyncTask 结果落库口径单元测试")
@ExtendWith(MockitoExtension.class)
class PredLogAsyncTaskTest {

    private static final String MODEL = "aecrnet";
    private static final String PRED_MD5 = "d41d8cd98f00b204e9800998ecf8427e";
    private static final String ORIGIN_MD5 = "0cc175b9c0f1b6a831c399e269772661";

    @Mock
    private SysPredLogMapper predLogMapper;

    @Mock
    private PythonAlgorithmClient pythonClient;

    @Test
    @DisplayName("pred_md5/origin_md5 取自算法服务响应")
    void writesMd5FromAlgorithmResponse() {
        JSONObject resp = new JSONObject();
        resp.set("taskId", 1);
        resp.set("status", LogStatusEnum.COMPLETED.getValue());
        resp.set("resultUrl", "http://cdn.test/result.png");
        resp.set("resultMd5", PRED_MD5);
        resp.set("originMd5", ORIGIN_MD5);
        when(pythonClient.predict(MODEL, null, "http://cdn.test/origin.png", null))
                .thenReturn(resp);

        PredLogAsyncTask task = new PredLogAsyncTask(predLogMapper, pythonClient, new SimpleMeterRegistry());
        task.execute(5L, MODEL, null, "http://cdn.test/origin.png", null);

        ArgumentCaptor<SysPredLog> captor = ArgumentCaptor.forClass(SysPredLog.class);
        verify(predLogMapper).updateById(captor.capture());
        SysPredLog updated = captor.getValue();
        assertThat(updated.getStatus()).isEqualTo(LogStatusEnum.COMPLETED);
        assertThat(updated.getPredUrl()).isEqualTo("http://cdn.test/result.png");
        assertThat(updated.getPredMd5()).isEqualTo(PRED_MD5);
        assertThat(updated.getOriginMd5()).isEqualTo(ORIGIN_MD5);
    }
}
