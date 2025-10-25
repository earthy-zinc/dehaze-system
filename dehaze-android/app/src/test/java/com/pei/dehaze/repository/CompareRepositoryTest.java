package com.pei.dehaze.repository;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.ArrayList;
import java.util.List;

@RunWith(MockitoJUnitRunner.class)
public class CompareRepositoryTest {

    private CompareRepository compareRepository;

    @Mock
    private CompareRepository.PredictionCallback predictionCallback;

    @Mock
    private CompareRepository.EvaluationCallback evaluationCallback;

    @Before
    public void setUp() {
        compareRepository = new CompareRepository();
    }

    @Test
    public void testGetPredictionSuccess() {
        // 模拟 ModelAPI.prediction 成功响应
        PredResult mockResult = new PredResult();
        mockResult.setPredUrl("http://example.com/pred.jpg");

        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.prediction(any(PredParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PredResult> callback = invocation.getArgument(1);
                        callback.onSuccess(mockResult);
                        return null;
                    });

            // 执行获取预测结果
            PredParam param = new PredParam();
            compareRepository.getPrediction(param, predictionCallback);

            // 验证回调被调用
            verify(predictionCallback).onSuccess(mockResult);
        }
    }

    @Test
    public void testGetPredictionError() {
        // 模拟 ModelAPI.prediction 错误响应
        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.prediction(any(PredParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PredResult> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取预测结果
            PredParam param = new PredParam();
            compareRepository.getPrediction(param, predictionCallback);

            // 验证回调被调用
            verify(predictionCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetPredictionNetworkFailure() {
        // 模拟 ModelAPI.prediction 网络失败
        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.prediction(any(PredParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<PredResult> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取预测结果
            PredParam param = new PredParam();
            compareRepository.getPrediction(param, predictionCallback);

            // 验证回调被调用
            verify(predictionCallback).onError("Network error: Network error");
        }
    }

    @Test
    public void testGetEvaluationSuccess() {
        // 模拟 ModelAPI.evaluation 成功响应
        List<EvalResult> mockResults = new ArrayList<>();
        EvalResult result1 = new EvalResult();
        result1.setId(1);
        result1.setLabel("PSNR");
        result1.setValue("30.5");
        result1.setDescription("峰值信噪比");
        mockResults.add(result1);

        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.evaluation(any(EvalParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<EvalResult>> callback = invocation.getArgument(1);
                        callback.onSuccess(mockResults);
                        return null;
                    });

            // 执行获取评估结果
            EvalParam param = new EvalParam();
            compareRepository.getEvaluation(param, evaluationCallback);

            // 验证回调被调用
            verify(evaluationCallback).onSuccess(mockResults);
        }
    }

    @Test
    public void testGetEvaluationError() {
        // 模拟 ModelAPI.evaluation 错误响应
        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.evaluation(any(EvalParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<EvalResult>> callback = invocation.getArgument(1);
                        callback.onError(404, "Not Found");
                        return null;
                    });

            // 执行获取评估结果
            EvalParam param = new EvalParam();
            compareRepository.getEvaluation(param, evaluationCallback);

            // 验证回调被调用
            verify(evaluationCallback).onError("Error 404: Not Found");
        }
    }

    @Test
    public void testGetEvaluationNetworkFailure() {
        // 模拟 ModelAPI.evaluation 网络失败
        try (MockedStatic<ModelAPI> mockedModelAPI = Mockito.mockStatic(ModelAPI.class)) {
            mockedModelAPI.when(() -> ModelAPI.evaluation(any(EvalParam.class), any()))
                    .thenAnswer(invocation -> {
                        ApiCallback<List<EvalResult>> callback = invocation.getArgument(1);
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(0, "Network error"));
                        return null;
                    });

            // 执行获取评估结果
            EvalParam param = new EvalParam();
            compareRepository.getEvaluation(param, evaluationCallback);

            // 验证回调被调用
            verify(evaluationCallback).onError("Network error: Network error");
        }
    }
}