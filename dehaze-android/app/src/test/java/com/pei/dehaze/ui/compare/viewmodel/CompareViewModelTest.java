package com.pei.dehaze.ui.compare.viewmodel;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;

import androidx.arch.core.executor.testing.InstantTaskExecutorRule;
import androidx.lifecycle.Observer;

import com.pei.dehaze.repository.CompareRepository;
import com.pei.dehaze.sdk.model.model.EvalParam;
import com.pei.dehaze.sdk.model.model.EvalResult;
import com.pei.dehaze.sdk.model.model.PredParam;
import com.pei.dehaze.sdk.model.model.PredResult;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;

@Config(sdk = 28)
@RunWith(RobolectricTestRunner.class)
public class CompareViewModelTest {

    // 确保 LiveData 在测试中立即执行
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();

    private CompareViewModel compareViewModel;

    @Mock
    private CompareRepository compareRepository;

    @Mock
    private Observer<PredResult> predictionResultObserver;

    @Mock
    private Observer<List<EvalResult>> evaluationResultsObserver;

    @Mock
    private Observer<Boolean> loadingObserver;

    @Mock
    private Observer<String> errorObserver;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        compareViewModel = new CompareViewModel();
        // 使用反射注入 mock 的 repository
        try {
            java.lang.reflect.Field field = CompareViewModel.class.getDeclaredField("compareRepository");
            field.setAccessible(true);
            field.set(compareViewModel, compareRepository);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testInitialState() {
        // 测试初始状态
        assertNotNull(compareViewModel.getPredictionResult());
        assertNotNull(compareViewModel.getEvaluationResults());
        assertNotNull(compareViewModel.getLoading());
        assertNotNull(compareViewModel.getError());

        assertNull(compareViewModel.getLoading().getValue());
        assertNull(compareViewModel.getError().getValue());
    }

    @Test
    public void testLoadPredictionSuccess() {
        // 模拟成功获取预测结果
        PredResult mockResult = new PredResult();
        mockResult.setPredUrl("http://example.com/pred.jpg");

        doAnswer(invocation -> {
            CompareRepository.PredictionCallback callback = invocation.getArgument(1);
            callback.onSuccess(mockResult);
            return null;
        }).when(compareRepository).getPrediction(any(PredParam.class), any());

        // 观察数据变化
        compareViewModel.getPredictionResult().observeForever(predictionResultObserver);
        compareViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载预测
        PredParam param = new PredParam();
        compareViewModel.loadPrediction(param);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(predictionResultObserver).onChanged(mockResult);
    }

    @Test
    public void testLoadPredictionError() {
        // 模拟获取预测结果失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            CompareRepository.PredictionCallback callback = invocation.getArgument(1);
            callback.onError(errorMessage);
            return null;
        }).when(compareRepository).getPrediction(any(PredParam.class), any());

        // 观察数据变化
        compareViewModel.getError().observeForever(errorObserver);
        compareViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载预测
        PredParam param = new PredParam();
        compareViewModel.loadPrediction(param);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }

    @Test
    public void testLoadEvaluationSuccess() {
        // 模拟成功获取评估结果
        List<EvalResult> mockResults = new ArrayList<>();
        EvalResult result1 = new EvalResult();
        result1.setId(1);
        result1.setLabel("PSNR");
        result1.setValue("30.5");
        result1.setDescription("峰值信噪比");
        mockResults.add(result1);

        doAnswer(invocation -> {
            CompareRepository.EvaluationCallback callback = invocation.getArgument(1);
            callback.onSuccess(mockResults);
            return null;
        }).when(compareRepository).getEvaluation(any(EvalParam.class), any());

        // 观察数据变化
        compareViewModel.getEvaluationResults().observeForever(evaluationResultsObserver);
        compareViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载评估
        EvalParam param = new EvalParam();
        compareViewModel.loadEvaluation(param);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(evaluationResultsObserver).onChanged(mockResults);
    }

    @Test
    public void testLoadEvaluationError() {
        // 模拟获取评估结果失败
        String errorMessage = "Network error";

        doAnswer(invocation -> {
            CompareRepository.EvaluationCallback callback = invocation.getArgument(1);
            callback.onError(errorMessage);
            return null;
        }).when(compareRepository).getEvaluation(any(EvalParam.class), any());

        // 观察数据变化
        compareViewModel.getError().observeForever(errorObserver);
        compareViewModel.getLoading().observeForever(loadingObserver);

        // 执行加载评估
        EvalParam param = new EvalParam();
        compareViewModel.loadEvaluation(param);

        // 验证状态变化
        verify(loadingObserver).onChanged(true);
        verify(loadingObserver).onChanged(false);
        verify(errorObserver).onChanged(errorMessage);
    }
}