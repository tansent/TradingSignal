// 通达信公式 DLL：按日期读取 D:\TradingSignal\data\tdx_nt.csv（由 daily/update.py 每天生成）。
// 文件每行：YYYYMMDD,净申购亿元,区间(1 入场 / -1 离场 / 0),当天(1 入场日 / -1 大额赎回日 / 0)
//
// 公式里的用法（绑定到第 1 号 DLL 时）：
//   TDXDLL1(1, DATE, 0, 0)  当天上交所宽基 ETF 净申购（亿元），没有数据时为 0
//   TDXDLL1(2, DATE, 0, 0)  国家队区间：1 入场区间、-1 离场区间、0 其他
//   TDXDLL1(3, DATE, 0, 0)  当天：1 国家队式入场日、-1 大额赎回日、0 其他
//   TDXDLL1(9, DATE, 0, 0)  原样返回输入（测试接口用）
// 通达信的 DATE 是 (年-1900)*10000+月*100+日，例如 2026-10-13 为 1261013。
// 文件修改后会自动重新读取。

#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <vector>

#pragma pack(push, 1)
typedef void (*pPluginFUNC)(int DataLen, float* pfOUT, float* pfINa, float* pfINb, float* pfINc);
typedef struct tagPluginTCalcFuncInfo {
    unsigned short nFuncMark;
    pPluginFUNC pCallFunc;
} PluginTCalcFuncInfo;
#pragma pack(pop)

namespace {

const char* kPath = "D:\\TradingSignal\\data\\tdx_nt.csv";

struct Row {
    int date;  // YYYYMMDD
    float net, zone, flag;
};

std::vector<Row> g_rows;
FILETIME g_mtime = {0, 0};
CRITICAL_SECTION g_lock;

void Reload() {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExA(kPath, GetFileExInfoStandard, &a)) {
        g_rows.clear();
        return;
    }
    if (CompareFileTime(&a.ftLastWriteTime, &g_mtime) == 0 && !g_rows.empty()) return;
    std::vector<Row> rows;
    FILE* f = nullptr;
    if (fopen_s(&f, kPath, "r") == 0 && f) {
        Row r;
        while (fscanf_s(f, "%d,%f,%f,%f", &r.date, &r.net, &r.zone, &r.flag) == 4) rows.push_back(r);
        fclose(f);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.date < y.date; });
    g_rows.swap(rows);
    g_mtime = a.ftLastWriteTime;
}

const Row* Find(float tdxDate) {
    int d = static_cast<int>(tdxDate + 0.5f) + 19000000;  // 1261013 -> 20261013
    auto it = std::lower_bound(g_rows.begin(), g_rows.end(), d, [](const Row& r, int v) { return r.date < v; });
    return (it != g_rows.end() && it->date == d) ? &*it : nullptr;
}

void Lookup(int n, float* out, const float* dates, int field) {
    EnterCriticalSection(&g_lock);
    Reload();
    for (int i = 0; i < n; i++) {
        const Row* r = Find(dates[i]);
        out[i] = r ? (field == 0 ? r->net : field == 1 ? r->zone : r->flag) : 0.0f;
    }
    LeaveCriticalSection(&g_lock);
}

void FuncNet(int n, float* out, float* a, float*, float*) { Lookup(n, out, a, 0); }
void FuncZone(int n, float* out, float* a, float*, float*) { Lookup(n, out, a, 1); }
void FuncFlag(int n, float* out, float* a, float*, float*) { Lookup(n, out, a, 2); }
void FuncEcho(int n, float* out, float* a, float*, float*) {
    for (int i = 0; i < n; i++) out[i] = a[i];
}

PluginTCalcFuncInfo g_funcs[] = {
    {1, &FuncNet}, {2, &FuncZone}, {3, &FuncFlag}, {9, &FuncEcho}, {0, nullptr},
};

}  // namespace

extern "C" __declspec(dllexport) BOOL RegisterTdxFunc(PluginTCalcFuncInfo** pFun) {
    if (*pFun == nullptr) {
        *pFun = g_funcs;
        return TRUE;
    }
    return FALSE;
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) InitializeCriticalSection(&g_lock);
    if (reason == DLL_PROCESS_DETACH) DeleteCriticalSection(&g_lock);
    return TRUE;
}
