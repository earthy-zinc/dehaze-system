package com.pei.dehaze.filter;

import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.ResponseUtils;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.impl.file.LocalFileService;
import jakarta.annotation.Resource;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@Slf4j
public class FileDownloadFilter implements Filter {
    @Resource
    private FileService fileService;

    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain) throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) servletRequest;
        HttpServletResponse httpResponse = (HttpServletResponse) servletResponse;
        String path = httpRequest.getRequestURI();

        if (path.startsWith("/api/v1/files/check")) {
            filterChain.doFilter(servletRequest, servletResponse);
        } else if (path.startsWith("/api/v1/files/")) {
            String filePath = path.substring("/api/v1/files/".length());
            if (fileService instanceof LocalFileService localFileService) {
                localFileService.download(filePath, httpRequest, httpResponse);
            } else {
                ResponseUtils.writeErrMsg(httpResponse, ResultCode.RESOURCE_NOT_FOUND);
            }
        } else {
            filterChain.doFilter(servletRequest, servletResponse); // 非目标路径，直接传递给下一个过滤器
        }
    }
}
