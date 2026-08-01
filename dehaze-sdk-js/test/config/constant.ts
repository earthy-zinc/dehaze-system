import dotenv from "dotenv";
import path from "path";

// 加载项目根 .env 到 process.env
dotenv.config({
  path: path.resolve(__dirname, "../../../.env"),
  quiet: true,
});

export const DEHAZE_HOST = process.env.DEHAZE_HOST || "127.0.0.1";
export const DEHAZE_PASSWORD = process.env.DEHAZE_PASSWORD || "Dehaze2026";
export const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8989";

const PORT_TO_BACKEND: Record<string, string> = {
  "8989": "dehaze-java",
  "8990": "dehaze-go",
  "8991": "dehaze-python",
};
export const BACKEND_NAME = PORT_TO_BACKEND[new URL(BACKEND_URL).port] || "dehaze-java";
