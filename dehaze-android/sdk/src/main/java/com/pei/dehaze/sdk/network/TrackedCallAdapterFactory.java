package com.pei.dehaze.sdk.network;

import java.lang.annotation.Annotation;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

import retrofit2.Call;
import retrofit2.CallAdapter;
import retrofit2.Retrofit;

/**
 * Retrofit CallAdapter 工厂：将每个 {@link Call} 包装为 {@link TrackedCall}，
 * 使请求可被登记到当前线程的 {@link CallTracker}，供 ViewModel 精确取消。
 *
 * <p>该工厂优先于 Retrofit 默认的 DefaultCallAdapterFactory 生效，仅处理 {@code Call<R>} 返回类型，
 * 不影响其它返回类型（如 Observable、LiveData 等，本 SDK 未使用）。
 */
public final class TrackedCallAdapterFactory extends CallAdapter.Factory {

    @Override
    public CallAdapter<?, ?> get(Type returnType, Annotation[] annotations, Retrofit retrofit) {
        if (getRawType(returnType) != Call.class) {
            return null;
        }
        if (!(returnType instanceof ParameterizedType)) {
            throw new IllegalStateException("Call 返回类型必须参数化为 Call<Foo>");
        }
        final Type responseType = ((ParameterizedType) returnType).getActualTypeArguments()[0];
        return new CallAdapter<Object, Call<?>>() {
            @Override
            public Type responseType() {
                return responseType;
            }

            @Override
            public Call<?> adapt(Call<Object> call) {
                return new TrackedCall<>(call);
            }
        };
    }
}
