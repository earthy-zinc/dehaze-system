package com.pei.dehaze.utils;

import android.content.Context;
import android.net.Uri;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * Uri 工具类，消除项目中重复的 Uri 转 File / 文件名查询方法。
 * 工具类只负责 IO，不弹 Toast；调用方自行处理 UI 提示。
 */
public final class UriUtils {
    private UriUtils() {}

    /** 将 Uri 内容复制为缓存文件，失败返回 null */
    public static File copyToCache(Context context, Uri uri) {
        if (context == null || uri == null) return null;
        try {
            InputStream is = context.getContentResolver().openInputStream(uri);
            if (is == null) return null;
            String fileName = getFileName(context, uri);
            File tempFile = new File(context.getCacheDir(), fileName != null ? fileName : "upload_temp");
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int len;
                while ((len = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }
            is.close();
            return tempFile;
        } catch (Exception e) {
            return null;
        }
    }

    /** 从 Uri 查询文件名，查询失败回退到 lastPathSegment */
    public static String getFileName(Context context, Uri uri) {
        if (uri == null) return null;
        String result = null;
        if ("content".equals(uri.getScheme()) && context != null) {
            try (android.database.Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    if (idx >= 0) {
                        result = cursor.getString(idx);
                    }
                }
            } catch (Exception ignored) {
            }
        }
        if (result == null) {
            result = uri.getLastPathSegment();
        }
        return result;
    }
}
