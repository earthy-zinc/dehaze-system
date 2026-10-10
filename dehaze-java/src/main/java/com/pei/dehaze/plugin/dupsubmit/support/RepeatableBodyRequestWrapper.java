package com.pei.dehaze.plugin.dupsubmit.support;

import jakarta.servlet.ReadListener;
import jakarta.servlet.ServletInputStream;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import org.springframework.util.StreamUtils;

/**
 * 可重复读取请求体的包装器。
 *
 * <p>防重切面是 {@code @Around}，执行时机在 {@code @RequestBody} 参数解析**之后**，
 * 此时原始请求流已被读空 —— 切面直接读 {@code getInputStream()} 只能拿到空串。
 * 因此必须由前置过滤器（{@link com.pei.dehaze.plugin.dupsubmit.filter.RepeatableBodyFilter}）
 * 先把请求体缓存下来，切面再从缓存取摘要。
 *
 * @author earthyzinc
 */
public class RepeatableBodyRequestWrapper extends HttpServletRequestWrapper {

    private final byte[] cachedBody;

    public RepeatableBodyRequestWrapper(HttpServletRequest request) throws IOException {
        super(request);
        this.cachedBody = StreamUtils.copyToByteArray(request.getInputStream());
    }

    /**
     * 缓存的原始请求体，供防重切面计算摘要。
     */
    public byte[] getCachedBody() {
        return cachedBody;
    }

    @Override
    public ServletInputStream getInputStream() {
        return new CachedBodyServletInputStream(cachedBody);
    }

    @Override
    public BufferedReader getReader() {
        Charset charset =
                getCharacterEncoding() != null ? Charset.forName(getCharacterEncoding()) : StandardCharsets.UTF_8;
        return new BufferedReader(new InputStreamReader(getInputStream(), charset));
    }

    /**
     * 基于字节数组的输入流：每次调用都返回可重读的新流，供参数解析与切面各读一次。
     */
    private static final class CachedBodyServletInputStream extends ServletInputStream {

        private final ByteArrayInputStream source;

        CachedBodyServletInputStream(byte[] body) {
            this.source = new ByteArrayInputStream(body);
        }

        @Override
        public int read() {
            return source.read();
        }

        @Override
        public int read(byte[] b, int off, int len) {
            return source.read(b, off, len);
        }

        @Override
        public boolean isFinished() {
            return source.available() == 0;
        }

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public void setReadListener(ReadListener readListener) {
            throw new UnsupportedOperationException("防重请求体缓存不支持异步读取");
        }
    }
}
