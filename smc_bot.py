"""
SMC Bot - quét tín hiệu Smart Money Concept và gửi về Telegram.
Chạy tự động bằng GitHub Actions (xem .github/workflows/smc.yml).
"""
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ===================== CẤU HÌNH (đổi trong file workflow) =====================
# SYMBOLS dạng "nguồn:mã", cách nhau bằng dấu phẩy. Nguồn: binance | mexc | yahoo
#   binance:CLUSDT  -> Binance Futures (dầu, crypto, cổ phiếu...)
#   mexc:SPX500_USDT -> MEXC Futures (chỉ số S&P 500)
SYMBOLS    = [s.strip() for s in os.getenv("SYMBOLS", "binance:CLUSDT,binance:BTCUSDT,mexc:SPX500_USDT").split(",") if s.strip()]
TF         = os.getenv("TF", "15m")                 # khung vào lệnh
HTF        = os.getenv("HTF", "4h")                 # khung lọc xu hướng
SWING      = int(os.getenv("SWING", "5"))           # độ dài swing (pivot)
NEED_SWEEP = os.getenv("NEED_SWEEP", "1") == "1"    # bắt buộc quét thanh khoản
NEED_FVG   = os.getenv("NEED_FVG", "0") == "1"      # bắt buộc có FVG
USE_HTF    = os.getenv("USE_HTF", "1") == "1"       # lọc xu hướng khung lớn
SWEEP_LB   = int(os.getenv("SWEEP_LB", "20"))       # số nến tìm sweep
OB_VALID   = int(os.getenv("OB_VALID", "50"))       # OB còn hiệu lực (số nến)
RR         = float(os.getenv("RR", "2"))            # tỷ lệ R:R
SL_BUF     = float(os.getenv("SL_BUF", "0.2"))      # đệm SL (x ATR)
RECENT     = int(os.getenv("RECENT", "3"))          # chỉ báo tín hiệu trong N nến gần nhất

TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
STATE_FILE = "state.json"

TF_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "8h": 28800, "1d": 86400}
MEXC_TF = {"1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30", "1h": "Min60",
           "4h": "Hour4", "8h": "Hour8", "1d": "Day1"}
HEADERS = {"User-Agent": "Mozilla/5.0 (smc-bot)"}


# ===================== LẤY DỮ LIỆU =====================
def _binance_klines(url, sym, limit):
    r = requests.get(url, params={"symbol": sym, "interval": TF, "limit": limit}, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    df = pd.DataFrame(r.json()).iloc[:, :5]
    df.columns = ["time", "open", "high", "low", "close"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time").astype(float)


def fetch_binance_futures(sym):
    return _binance_klines("https://fapi.binance.com/fapi/v1/klines", sym, 1500)


def fetch_binance_spot(sym):
    return _binance_klines("https://data-api.binance.vision/api/v3/klines", sym, 1000)


def fetch_mexc(sym):
    if TF not in MEXC_TF:
        raise RuntimeError(f"MEXC không hỗ trợ khung {TF}")
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - TF_SEC[TF] * 1500
    last_err = None
    for host in ["https://contract.mexc.com", "https://api.mexc.com"]:
        try:
            r = requests.get(f"{host}/api/v1/contract/kline/{sym}",
                             params={"interval": MEXC_TF[TF], "start": start, "end": end},
                             headers=HEADERS, timeout=20)
            js = r.json()
            d = js.get("data") or {}
            if not js.get("success") or not d.get("time"):
                raise RuntimeError(f"HTTP {r.status_code}: {str(js)[:120]}")
            df = pd.DataFrame({k: d[k] for k in ["time", "open", "high", "low", "close"]})
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            return df.set_index("time").astype(float)
        except Exception as e:
            last_err = e
    raise RuntimeError(last_err)


def fetch_yahoo(sym):
    import yfinance as yf
    period = "60d" if TF.endswith("m") else "730d"
    df = yf.download(sym, period=period, interval=TF, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def to_mexc(sym):
    """BTCUSDT -> BTC_USDT"""
    return sym if "_" in sym else (sym[:-4] + "_USDT" if sym.endswith("USDT") else sym)


def fetch(item):
    """Trả về (dataframe nến đã đóng, tên nguồn đã dùng)."""
    src, sym = item.split(":", 1) if ":" in item else ("binance", item)
    src = src.lower()
    if src == "binance":
        # Binance Futures; nếu bị chặn (vd. lỗi 451 do máy chủ GitHub ở Mỹ) thì tự chuyển nguồn dự phòng
        tries = [("Binance Futures", fetch_binance_futures, sym),
                 ("MEXC Futures (dự phòng)", fetch_mexc, to_mexc(sym)),
                 ("Binance Spot (dự phòng)", fetch_binance_spot, sym)]
    elif src == "mexc":
        tries = [("MEXC Futures", fetch_mexc, sym)]
    else:
        tries = [("Yahoo", fetch_yahoo, sym)]
    errors = []
    for name, fn, s in tries:
        try:
            df = fn(s)
            if len(df) > 100:
                return df.sort_index().iloc[:-1], name  # bỏ nến đang chạy
            errors.append(f"{name}: ít dữ liệu")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(" | ".join(errors))


# ===================== CẤU TRÚC THỊ TRƯỜNG =====================
def structure(h, l, c, n):
    """Trả về trend, brk (1 BOS tăng, 2 CHoCH tăng, -1 BOS giảm, -2 CHoCH giảm), mức đỉnh/đáy swing."""
    N = len(c)
    trend = np.zeros(N, int)
    brk = np.zeros(N, int)
    hi = np.full(N, np.nan)
    lo = np.full(N, np.nan)
    hi_lvl = lo_lvl = np.nan
    hi_done = lo_done = True
    t = 0
    for i in range(N):
        p = i - n  # pivot được xác nhận sau n nến
        if p - n >= 0:
            wh, wl = h[p - n:p + n + 1], l[p - n:p + n + 1]
            if h[p] >= wh.max() and h[p] > h[p - n:p].max():
                hi_lvl, hi_done = h[p], False
            if l[p] <= wl.min() and l[p] < l[p - n:p].min():
                lo_lvl, lo_done = l[p], False
        b = 0
        if not hi_done and c[i] > hi_lvl:
            b = 1 if t == 1 else 2
            t, hi_done = 1, True
        elif not lo_done and c[i] < lo_lvl:
            b = -1 if t == -1 else -2
            t, lo_done = -1, True
        trend[i], brk[i], hi[i], lo[i] = t, b, hi_lvl, lo_lvl
    return trend, brk, hi, lo


def htf_trend(df):
    """Xu hướng khung lớn, chỉ dùng nến HTF đã đóng (không repaint)."""
    if not USE_HTF:
        return np.zeros(len(df), int)
    rule = HTF.replace("m", "min") if HTF.endswith("m") else HTF
    hdf = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    t, _, _, _ = structure(hdf.high.values, hdf.low.values, hdf.close.values, SWING)
    ht = pd.Series(t, index=hdf.index).shift(1)  # nến HTF trước đó (đã đóng)
    return ht.reindex(df.index, method="ffill").fillna(0).astype(int).values


# ===================== TÌM TÍN HIỆU =====================
def find_signals(df):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    N = len(c)
    trend, brk, hi, lo = structure(h, l, c, SWING)
    ht = htf_trend(df)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(14).mean().values

    last_sweep_bull = last_sweep_bear = last_fvg_bull = last_fvg_bear = -10**9
    b_arm = s_arm = False
    b_top = b_bot = s_top = s_bot = np.nan
    b_bar = s_bar = 0
    signals = []

    for i in range(2, N):
        # Quét thanh khoản & FVG
        if l[i] < lo[i - 1] and c[i] > lo[i - 1]:
            last_sweep_bull = i
        if h[i] > hi[i - 1] and c[i] < hi[i - 1]:
            last_sweep_bear = i
        if l[i] > h[i - 2]:
            last_fvg_bull = i
        if h[i] < l[i - 2]:
            last_fvg_bear = i

        bull_ok = (not USE_HTF) or ht[i] == 1
        bear_ok = (not USE_HTF) or ht[i] == -1

        # Phá cấu trúc -> đánh dấu Order Block
        if brk[i] > 0:
            s_arm = False
            if (not NEED_SWEEP or i - last_sweep_bull <= SWEEP_LB) and \
               (not NEED_FVG or i - last_fvg_bull <= SWING * 2) and bull_ok:
                for k in range(1, 31):
                    if i - k >= 0 and c[i - k] < o[i - k]:
                        b_arm, b_top, b_bot, b_bar = True, h[i - k], l[i - k], i
                        break
        elif brk[i] < 0:
            b_arm = False
            if (not NEED_SWEEP or i - last_sweep_bear <= SWEEP_LB) and \
               (not NEED_FVG or i - last_fvg_bear <= SWING * 2) and bear_ok:
                for k in range(1, 31):
                    if i - k >= 0 and c[i - k] > o[i - k]:
                        s_arm, s_top, s_bot, s_bar = True, h[i - k], l[i - k], i
                        break

        # Giá hồi về OB -> vào lệnh
        if b_arm and i > b_bar:
            if c[i] < b_bot or i - b_bar > OB_VALID:
                b_arm = False
            elif l[i] <= b_top and c[i] > b_bot and bull_ok:
                sl = b_bot - atr[i] * SL_BUF
                signals.append((i, "BUY", c[i], sl, c[i] + (c[i] - sl) * RR))
                b_arm = False
        if s_arm and i > s_bar:
            if c[i] > s_top or i - s_bar > OB_VALID:
                s_arm = False
            elif h[i] >= s_bot and c[i] < s_top and bear_ok:
                sl = s_top + atr[i] * SL_BUF
                signals.append((i, "SELL", c[i], sl, c[i] - (sl - c[i]) * RR))
                s_arm = False
    return signals


# ===================== TELEGRAM & TRẠNG THÁI =====================
def send(text):
    if not TOKEN or not CHAT_ID:
        print("[Thiếu TOKEN/CHAT_ID]\n" + text)
        return
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    print("Telegram:", r.status_code, r.text[:200])


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(x):
    return f"{x:.5f}" if abs(x) < 10 else f"{x:.2f}"


def main():
    if os.getenv("SEND_TEST") == "1":
        send("✅ SMC Bot kết nối Telegram thành công!")
    state = load_state()
    for item in SYMBOLS:
        sym = item.split(":", 1)[-1]
        try:
            df, source = fetch(item)
            sigs = find_signals(df)
            last_sent = state.get(item, "")
            for i, side, entry, sl, tp in sigs:
                t = df.index[i].isoformat()
                if i >= len(df) - RECENT and t > last_sent:
                    icon = "🟢" if side == "BUY" else "🔴"
                    send(f"{icon} SMC {side} | {sym} | TF {TF}\n"
                         f"Nguồn: {source}\n"
                         f"Thời gian: {df.index[i].strftime('%d/%m %H:%M')} UTC\n"
                         f"Entry: {fmt(entry)}\nSL: {fmt(sl)}\nTP: {fmt(tp)}\nR:R 1:{RR:g}")
                    state[item] = t
            print(f"{item}: lấy từ {source}, {len(df)} nến, {len(sigs)} tín hiệu trong lịch sử.")
        except Exception as e:
            print(f"{item} LỖI -> {e}")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()
