// 通达信公式 DLL：按日期读取 daily/update.py 每天生成的数据文件。
//   D:\TradingSignal\data\tdx_nt.csv     每行：YYYYMMDD,净申购亿元,区间(1 入场 / -1 离场 / 0),当天(1 入场日 / -1 大额赎回日 / 0)
//   D:\TradingSignal\data\tdx_crowd.csv  每行：YYYYMMDD,拥挤度%,区域(2 高危 / 1 警戒 / 0),◆S(1 / 0)
//   D:\TradingSignal\data\tdx_margin.csv 每行：YYYYMMDD,融资余额20日变化%,融资余额亿,0（收盘时已知，即前一天的数）
//   D:\TradingSignal\data\tdx_extra.csv  通用表，每行：YYYYMMDD,第1列,...,第8列（列的含义见 research/icepoint.py）
//
// 公式里的用法（绑定到第 1 号 DLL 时）：
//   TDXDLL1(1, DATE, 0, 0)  当天上交所宽基 ETF 净申购（亿元），没有数据时为 0
//   TDXDLL1(2, DATE, 0, 0)  国家队区间：1 入场区间、-1 离场区间、0 其他
//   TDXDLL1(3, DATE, 0, 0)  当天：1 国家队式入场日、-1 大额赎回日、0 其他
//   TDXDLL1(4, DATE, 0, 0)  A股拥挤度（成交额前 5% 个股的成交额占比，%），没有数据时为 0
//   TDXDLL1(5, DATE, 0, 0)  抱团区域：2 高危（拥挤度≥48% 且沪深300 在一年高点 5% 内）、1 警戒（≥45%）、0 其他
//   TDXDLL1(6, DATE, 0, 0)  ◆S：每一轮高危区的第一天为 1
//   TDXDLL1(7, DATE, 0, 0)  融资余额 20 日变化（%，收盘时已知的前一天的数，来自 data\tdx_margin.csv），没有数据时为 0
//   TDXDLL1(8, DATE, 0, 0)  融资余额（亿元，前一天的数）
//   TDXDLL1(9, DATE, 0, 0)  原样返回输入（测试接口用）
//   TDXDLL1(21~28, DATE, 0, 0)  通用表 data\tdx_extra.csv 的第 1~8 列，没有数据时为 0；目前是：
//     21 全市场涨停家数、22 跌停家数、23 短线冰点（跌停>涨停 且 跌停≥20 为 1）、24 中线温度、25 短线温度分位
//     以后给公式加新的全市场数据，只需 Python 多写一列，不用重新编译 DLL
//   TDXDLL1(31~33, DATE, 估值号, 0)  估值通道 data\tdx_val_<估值号>.csv（research/valuation.py 写出），没有数据时为 0：
//     31 低分位比例、32 高分位比例（通道价位 = 收盘价 × 比例）、33 当前估值分位（0~100，当时可知）
//     估值号：1 上证指数、16 上证50、300 沪深300、905 中证500、852 中证1000、399001 深证成指、399006 创业板指、688 科创50、30269 红利低波
// 通达信的 DATE 是 (年-1900)*10000+月*100+日，例如 2026-10-13 为 1261013。
// 文件修改后会自动重新读取。

#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#pragma pack(push, 1)
typedef void (*pPluginFUNC)(int DataLen, float* pfOUT, float* pfINa, float* pfINb, float* pfINc);
typedef struct tagPluginTCalcFuncInfo {
    unsigned short nFuncMark;
    pPluginFUNC pCallFunc;
} PluginTCalcFuncInfo;
#pragma pack(pop)

namespace {

const int kCols = 8;

struct Row {
    int date;  // YYYYMMDD
    float v[kCols];
};

struct Table {
    std::string path;
    std::vector<Row> rows;
    FILETIME mtime;
};

Table g_nt = {"D:\\TradingSignal\\data\\tdx_nt.csv", {}, {0, 0}};
Table g_crowd = {"D:\\TradingSignal\\data\\tdx_crowd.csv", {}, {0, 0}};
Table g_margin = {"D:\\TradingSignal\\data\\tdx_margin.csv", {}, {0, 0}};
Table g_extra = {"D:\\TradingSignal\\data\\tdx_extra.csv", {}, {0, 0}};
CRITICAL_SECTION g_lock;

void Reload(Table& t) {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExA(t.path.c_str(), GetFileExInfoStandard, &a)) {
        t.rows.clear();
        return;
    }
    if (CompareFileTime(&a.ftLastWriteTime, &t.mtime) == 0 && !t.rows.empty()) return;
    std::vector<Row> rows;
    FILE* f = nullptr;
    if (fopen_s(&f, t.path.c_str(), "r") == 0 && f) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            Row r = {};
            char* p = line;
            r.date = static_cast<int>(strtol(p, &p, 10));
            for (int k = 0; k < kCols && *p == ','; k++) r.v[k] = strtof(p + 1, &p);
            if (r.date > 19000000) rows.push_back(r);
        }
        fclose(f);
    }
    std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.date < y.date; });
    t.rows.swap(rows);
    t.mtime = a.ftLastWriteTime;
}

const Row* Find(const Table& t, float tdxDate) {
    int d = static_cast<int>(tdxDate + 0.5f) + 19000000;  // 1261013 -> 20261013
    auto it = std::lower_bound(t.rows.begin(), t.rows.end(), d, [](const Row& r, int v) { return r.date < v; });
    return (it != t.rows.end() && it->date == d) ? &*it : nullptr;
}

void Lookup(Table& t, int n, float* out, const float* dates, int field) {
    EnterCriticalSection(&g_lock);
    Reload(t);
    for (int i = 0; i < n; i++) {
        const Row* r = Find(t, dates[i]);
        out[i] = r ? r->v[field] : 0.0f;
    }
    LeaveCriticalSection(&g_lock);
}

void FuncNet(int n, float* out, float* a, float*, float*) { Lookup(g_nt, n, out, a, 0); }
void FuncZone(int n, float* out, float* a, float*, float*) { Lookup(g_nt, n, out, a, 1); }
void FuncFlag(int n, float* out, float* a, float*, float*) { Lookup(g_nt, n, out, a, 2); }
void FuncCrowd(int n, float* out, float* a, float*, float*) { Lookup(g_crowd, n, out, a, 0); }
void FuncCrowdZone(int n, float* out, float* a, float*, float*) { Lookup(g_crowd, n, out, a, 1); }
void FuncCrowdMark(int n, float* out, float* a, float*, float*) { Lookup(g_crowd, n, out, a, 2); }
void FuncMarginChg(int n, float* out, float* a, float*, float*) { Lookup(g_margin, n, out, a, 0); }
void FuncMarginBal(int n, float* out, float* a, float*, float*) { Lookup(g_margin, n, out, a, 1); }
template <int K>
void FuncExtra(int n, float* out, float* a, float*, float*) { Lookup(g_extra, n, out, a, K); }
// 估值通道：每个指数一个文件 data\tdx_val_<编号>.csv，编号由公式的第 3 个参数传入（第一次用到时才读）
std::map<int, Table> g_val;

template <int K>
void FuncVal(int n, float* out, float* a, float* b, float*) {
    int id = n > 0 ? static_cast<int>(b[n - 1] + 0.5f) : 0;
    if (id <= 0) {
        for (int i = 0; i < n; i++) out[i] = 0.0f;
        return;
    }
    EnterCriticalSection(&g_lock);
    auto it = g_val.find(id);
    if (it == g_val.end()) {
        char path[128];
        sprintf_s(path, "D:\\TradingSignal\\data\\tdx_val_%d.csv", id);
        it = g_val.emplace(id, Table{path, {}, {0, 0}}).first;
    }
    Lookup(it->second, n, out, a, K);       // 临界区可重入
    LeaveCriticalSection(&g_lock);
}

void FuncEcho(int n, float* out, float* a, float*, float*) {
    for (int i = 0; i < n; i++) out[i] = a[i];
}

PluginTCalcFuncInfo g_funcs[] = {
    {1, &FuncNet}, {2, &FuncZone}, {3, &FuncFlag},
    {4, &FuncCrowd}, {5, &FuncCrowdZone}, {6, &FuncCrowdMark},
    {7, &FuncMarginChg}, {8, &FuncMarginBal},
    {9, &FuncEcho},
    {21, &FuncExtra<0>}, {22, &FuncExtra<1>}, {23, &FuncExtra<2>}, {24, &FuncExtra<3>},
    {25, &FuncExtra<4>}, {26, &FuncExtra<5>}, {27, &FuncExtra<6>}, {28, &FuncExtra<7>},
    {31, &FuncVal<0>}, {32, &FuncVal<1>}, {33, &FuncVal<2>},
    {0, nullptr},
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
