# Anaconda 環境設定指南

本指南說明如何使用 Anaconda 環境來運行 UI Agent 系統。

## 為什麼使用 Anaconda？

✅ **隔離環境**：不會影響系統的其他 Node.js 安裝  
✅ **版本控制**：確保所有人使用相同版本的 Node.js 和工具  
✅ **簡單管理**：一個命令就能設定好所有環境  
✅ **跨平台**：Windows、macOS、Linux 都能使用

## 🚀 快速開始（三步驟）

### 1️⃣ 安裝 Anaconda（如果還沒有）

訪問 https://docs.conda.io/en/latest/miniconda.html 下載並安裝 Miniconda

### 2️⃣ 設定環境（僅需一次）

```bash
./setup-conda.sh
```

### 3️⃣ 啟動系統

```bash
./start-with-conda.sh
```

完成！系統將自動啟動並可在以下地址訪問：
- Web UI: http://localhost:3000
- API Server: http://localhost:3001

## 📋 詳細說明

### environment.yml 檔案內容

```yaml
name: uiagent              # 環境名稱
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11           # Python 版本（供未來擴展）
  - nodejs=20.10.0        # Node.js 版本
  - npm=10.2.3            # npm 版本
  - pip                   # pip 套件管理器
  - pip:
    - pnpm                # pnpm 套件管理器
```

### setup-conda.sh 腳本功能

1. ✅ 檢查 conda 是否已安裝
2. ✅ 建立 `uiagent` conda 環境
3. ✅ 安裝 Node.js、npm 和其他依賴
4. ✅ 提供重新建立環境的選項

### start-with-conda.sh 腳本功能

1. ✅ 檢查並啟動 conda 環境
2. ✅ 驗證 Node.js 和 npm 安裝
3. ✅ 自動安裝 npm 依賴（首次執行）
4. ✅ 建置所有 packages
5. ✅ 建立 API 環境變數檔案
6. ✅ 啟動 API 和 Web 服務

## 🔧 常用命令

### 手動管理環境

```bash
# 啟動環境
conda activate uiagent

# 檢查環境中的 Node.js 版本
node --version

# 檢查 npm 版本
npm --version

# 離開環境
conda deactivate

# 列出所有環境
conda env list

# 刪除環境
conda env remove -n uiagent
```

### 在環境中手動操作

```bash
# 啟動環境
conda activate uiagent

# 安裝依賴
npm install

# 建置 packages
npm run build

# 啟動開發服務
npm run dev

# 離開環境
conda deactivate
```

## ❓ 常見問題

### Q: 如何更新 Node.js 版本？

**A:** 編輯 `environment.yml`，修改 `nodejs` 版本，然後重新建立環境：
```bash
conda env remove -n uiagent
./setup-conda.sh
```

### Q: 環境建立失敗怎麼辦？

**A:** 常見解決方法：
1. 確認網路連接正常
2. 更新 conda: `conda update conda`
3. 清除 conda 快取: `conda clean --all`
4. 重試建立環境

### Q: 如何在不同專案間切換？

**A:** 使用 conda 環境切換：
```bash
conda deactivate          # 離開當前環境
conda activate uiagent    # 啟動 UI Agent 環境
conda activate other-env  # 切換到其他環境
```

### Q: Windows 上如何執行 .sh 腳本？

**A:** 使用 Git Bash 或 WSL (Windows Subsystem for Linux)：
```bash
# Git Bash
bash ./setup-conda.sh
bash ./start-with-conda.sh

# 或使用 WSL
./setup-conda.sh
./start-with-conda.sh
```

### Q: 我可以不用腳本手動設定嗎？

**A:** 可以，執行以下命令：
```bash
# 建立環境
conda env create -f environment.yml

# 啟動環境
conda activate uiagent

# 安裝依賴並啟動
npm install
npm run build
npm run dev
```

## 🎯 最佳實踐

1. **每次開發前啟動環境**
   ```bash
   conda activate uiagent
   ```

2. **使用腳本簡化流程**
   ```bash
   ./start-with-conda.sh
   ```

3. **完成工作後離開環境**
   ```bash
   conda deactivate
   ```

4. **定期更新依賴**
   ```bash
   conda activate uiagent
   npm update
   ```

## 📚 延伸閱讀

- [Conda 官方文檔](https://docs.conda.io/)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Node.js 版本管理](https://nodejs.org/en/about/releases/)

## 🆘 需要幫助？

如果遇到任何問題：

1. 查看終端的錯誤訊息
2. 參考本文檔的「常見問題」章節
3. 檢查 [INSTALLATION.md](INSTALLATION.md) 完整安裝指南
4. 提交 Issue 到專案 repository
