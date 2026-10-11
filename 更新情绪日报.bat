@echo off
chcp 65001 >nul
title A股情绪日报 - 手动更新
cd /d D:\TradingSignal
echo.
echo [1/2] 正在重算全部指标（约 1 分钟）...
echo       通达信没有最近一天的数据时，会自动用新浪收盘快照补上，不需要打开通达信。
echo       如果好几天没运行，下面会出现「警告」：那时打开通达信做一次盘后数据下载，再运行本程序。
set PYTHONIOENCODING=utf-8
python daily\update.py
if errorlevel 1 (
  echo.
  echo 计算出错，请把上面的信息截图发给 Claude。
  pause
  exit /b 1
)
echo.
echo [2/2] 生成本地网页版日报...
python daily\render_local.py
start "" "D:\TradingSignal\reports\A股情绪日报.html"
echo.
echo 完成。本地日报已在浏览器打开。
echo 手机上的 claude.ai 看板需要由 Claude 同步：在 Claude 桌面应用的定时任务里对"A股情绪日报 每日更新"点"立即运行"。
echo.
pause
