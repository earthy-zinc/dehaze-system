import { app, BrowserWindow, ipcMain, Menu } from "electron";
import { resolve } from "node:path";

let mainWindow: BrowserWindow | null = null;

const createWindow = () => {
  const icon = resolve(
    __dirname,
    app.isPackaged ? "../renderer/favicon.ico" : "../../public/favicon.ico"
  );
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 600,
    frame: false,
    icon,
    backgroundColor: "#fff",
    webPreferences: {
      preload: resolve(__dirname, "../preload/index.mjs"),
      sandbox: false,
    },
  });

  if (!app.isPackaged && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(resolve(__dirname, "../../out/renderer/index.html"));
  }

  if (!app.isPackaged) mainWindow.webContents.openDevTools({ mode: "detach" });
};

ipcMain.on("window:minimize", () => mainWindow?.minimize());
ipcMain.on("window:toggleMaximize", () => {
  if (!mainWindow) return;
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});
ipcMain.on("window:close", () => mainWindow?.close());

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
