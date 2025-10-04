import { app, BrowserWindow } from "electron";
import path from "node:path";

const isDevelopment = process.env.NODE_ENV === "development";
let mainWindow = null;

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, "preload"),
    },
  });
  if (isDevelopment) {
    mainWindow.loadURL("http://localhost:3000");
  } else {
    mainWindow.loadFile("../dist/index.html");
  }
};
app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
