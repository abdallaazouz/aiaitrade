# -*- coding: utf-8 -*-
import os
import sys
import time
import io
import queue
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque

import numpy as np
import pandas as pd
import requests
import joblib
from scipy import stats as sp_stats
from iqoptionapi.stable_api import IQ_Option

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

try:
    from statsmodels.tsa.holtwinters import Holt
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def call_with_timeout(fn, timeout_sec=25):
    box = {"res": None, "err": None}
    def _run():
        try:
            box["res"] = fn()
        except Exception as exc:
            box["err"] = exc
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise TimeoutError(f"انتهت مهلة الانتظار ({timeout_sec} ثانية)")
    if box["err"] is not None:
        raise box["err"]
    return box["res"]

# ===========================================================================
# 1. الإعدادات من GitHub Secrets
# ===========================================================================
IQ_EMAIL = os.environ.get("IQ_EMAIL", "").strip()
IQ_PASSWORD = os.environ.get("IQ_PASSWORD", "")
APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL", "").strip()
SECRET_TOKEN = os.environ.get("SECRET_TOKEN", "").strip()
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "16200"))
ENABLE_KRONOS = os.environ.get("ENABLE_KRONOS", "1") == "1"

CAIRO_TZ = ZoneInfo("Africa/Cairo")
FOREX_PAIRS = frozenset({
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
})
DEFAULT_PAIR = os.environ.get("DEFAULT_PAIR", "EURUSD").strip().upper()

DATA_DIR = Path("data")
MODEL_VERSION = "v9-full-history"
MODEL_DIR = DATA_DIR / "models_v9"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DB = DATA_DIR / "iq_ai_data.sqlite3"

INTERVAL = 60
MAX_CANDLES_IN_MEMORY = 40000
MAX_TIMEFRAME_CANDLES = 1000
SEQUENCE_LENGTH = 64
MIN_TIMEFRAME_BARS_FOR_TRAINING = 300
MIN_DL_TRAINING_ROWS = 600
RETRAIN_NEW_CANDLES = 60
MIN_CANDLES_FOR_TRAINING = 650
TIMEFRAMES = (5, 10, 15, 30)
INITIAL_HISTORY_CANDLES = MAX_TIMEFRAME_CANDLES * max(TIMEFRAMES)
HISTORY_CHUNK_SIZE = 1000
STEPS = (1, 2, 3)

FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
    "ema5", "ema10", "ema20", "ema50", "ema100",
    "sma_cross_50_200", "ema_cross_20_100",
    "gma5", "gma10", "gma20", "gma50", "gma_cross_5_20", "gma_cross_10_50",
    "geo_ret5", "geo_ret15", "log_slope20", "log_r2_20", "log_slope50", "ou_zscore",
    "rsi7", "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "atr14", "stoch_k", "stoch_d",
    "cci20", "williams_r", "body", "upper_shadow",
    "lower_shadow", "range_pct", "doji", "hammer",
    "shooting_star", "bullish_engulfing", "bearish_engulfing",
    "vol_z", "ret_vol", "mom10", "mom20", "roc10",
    "trend_strength", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "adx14", "plus_di", "minus_di", "atr_ratio", "bb_squeeze",
    "dist_high20", "dist_low20", "dist_high60", "dist_low60",
    "streak", "rsi_slope", "macd_slope", "sma_cross_5_20", "ema_cross_10_50",
]

class FusionGBDTBrain:
    def __init__(self, gbdt_members, usable=True, quality_passed=True):
        self.gbdt_members = gbdt_members
        self.usable = usable
        self.quality_passed = quality_passed

    def predict_gbdt(self, X):
        if not self.usable:
            return None
        tot_w, acc = 0.0, 0.0
        for _, w, est in self.gbdt_members:
            p_all = est.predict_proba(X)
            classes = list(est.classes_)
            if 1 in classes:
                acc += w * float(p_all[0, classes.index(1)])
                tot_w += w
        return (acc / tot_w) if tot_w > 0 else 0.5

if TORCH_AVAILABLE:
    class DeepCandleSequenceNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.sequence = nn.GRU(n_features, 64, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(64), nn.Linear(64, 24), nn.SiLU(),
                nn.Dropout(0.15), nn.Linear(24, 1)
            )

        def forward(self, x):
            _, hidden = self.sequence(x)
            return self.head(hidden[-1]).squeeze(1)

stop_event = threading.Event()
train_queue = queue.Queue()
db_init_lock = threading.Lock()
db_initialized = False

# متابعة مهام التدريب ومنع إضافتها أكثر من مرة
training_jobs = set()
training_lock = threading.Lock()
active_training = None

gbdt_models, dl_models = {}, {}
last_mistake_retrain, last_retrain_ts = {}, {}
last_train_ts = {}

tg_config = {"enabled": False, "min_conf": 62, "cooldown_min": 60, "candles": ["5_1", "5_2", "5_3"]}
tg_cooldown_until = {}
tg_last_info = "لم يتم الإرسال بعد"

logs_buffer = deque(maxlen=40)
pending_sheet_preds = []
state_lock = threading.Lock()
latest_forecast = {"ok": False, "note": "جاري تهيئة النظام..."}
current_pair = DEFAULT_PAIR if DEFAULT_PAIR in FOREX_PAIRS else "EURUSD"

kronos_predictor = None
kronos_cache = {}  # (pair, last_ts) -> {(tf, step): prob}

def add_log(msg):
    line = f"[{datetime.now(CAIRO_TZ).strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    with state_lock:
        logs_buffer.append(line)

# ===========================================================================
# 2. تحميل وتشغيل نموذج Kronos (Foundation Model)
# ===========================================================================
def init_kronos():
    global kronos_predictor
    if not ENABLE_KRONOS:
        return
    repo_dir = Path("kronos_repo")
    if not repo_dir.exists():
        add_log("Kronos غير مستنسخ — سيعمل النظام بالمحركات الـ 4 الأخرى.")
        return
    try:
        sys.path.insert(0, str(repo_dir.resolve()))
        from model import Kronos, KronosTokenizer, KronosPredictor
        add_log("جاري تحميل Kronos-mini من HuggingFace...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        tokenizer.eval()
        model.eval()
        kronos_predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        add_log("✅ تم تفعيل نموذج Kronos بنجاح.")
    except Exception as exc:
        kronos_predictor = None
        add_log(f"تعذر تفعيل Kronos (سيكمل بدون تأثير): {exc}")

def kronos_predict_all(raw: pd.DataFrame, pair: str):
    """يتنبأ بـ 90 دقيقة للأمام بضربة واحدة ويحسب احتمال الصعود لكل من الـ 12 شمعة."""
    if kronos_predictor is None or raw is None or len(raw) < 120:
        return {}
    last_ts = int(raw.iloc[-1]["ts"])
    if (pair, last_ts) in kronos_cache:
        return kronos_cache[(pair, last_ts)]
    try:
        sub = raw.tail(512).copy().reset_index(drop=True)
        x_df = sub[["open", "high", "low", "close", "volume"]].astype(float)
        x_ts = pd.to_datetime(sub["ts"], unit="s")
        y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + pd.Timedelta(minutes=1), periods=90, freq="1min"))
        pred_df = kronos_predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=90, T=1.0, top_p=0.9, sample_count=3, verbose=False
        )
        closes = pred_df["close"].to_numpy(dtype=float)
        ref_close = float(sub.iloc[-1]["close"])
        vol = float(np.std(np.diff(np.log(np.maximum(sub["close"].to_numpy()[-60:], 1e-9))))) + 1e-6

        out = {}
        for tf in TIMEFRAMES:
            for step in STEPS:
                idx0 = (step - 1) * tf - 1
                idx1 = step * tf - 1
                p0 = ref_close if idx0 < 0 else closes[min(idx0, len(closes) - 1)]
                p1 = closes[min(idx1, len(closes) - 1)]
                log_ret = np.log(max(p1, 1e-9) / max(p0, 1e-9))
                z = log_ret / (vol * np.sqrt(tf) + 1e-6)
                out[(tf, step)] = float(sp_stats.norm.cdf(np.clip(z * 1.3, -2.5, 2.5)))
        kronos_cache.clear()
        kronos_cache[(pair, last_ts)] = out
        return out
    except Exception as exc:
        add_log(f"تنبيه Kronos: {exc}")
        return {}

# ===========================================================================
# 3. قاعدة البيانات
# ===========================================================================
def db_connect():
    global db_initialized
    c = sqlite3.connect(LOCAL_DB, timeout=60)
    c.execute("PRAGMA synchronous=NORMAL")
    if not db_initialized:
        with db_init_lock:
            if not db_initialized:
                try:
                    c.execute("PRAGMA journal_mode=WAL")
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS candles(
                            pair TEXT NOT NULL, ts INTEGER NOT NULL,
                            open REAL, high REAL, low REAL, close REAL, volume REAL,
                            PRIMARY KEY(pair, ts)
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS model_training(
                            pair TEXT NOT NULL, timeframe INTEGER NOT NULL,
                            step_idx INTEGER NOT NULL, model_version TEXT NOT NULL,
                            source_minute_candles INTEGER NOT NULL,
                            timeframe_candles INTEGER NOT NULL,
                            training_rows INTEGER NOT NULL,
                            quality_passed INTEGER NOT NULL,
                            trained_at TEXT NOT NULL,
                            dl_status TEXT NOT NULL DEFAULT 'pending',
                            dl_test_balanced_accuracy REAL,
                            dl_training_rows INTEGER NOT NULL DEFAULT 0,
                            dl_trained_at TEXT,
                            dl_error TEXT,
                            PRIMARY KEY(pair, timeframe, step_idx, model_version)
                        )
                    """)
                    training_cols = {
                        row[1] for row in c.execute("PRAGMA table_info(model_training)").fetchall()
                    }
                    for col_name, col_type in [
                        ("dl_status", "TEXT NOT NULL DEFAULT 'pending'"),
                        ("dl_test_balanced_accuracy", "REAL"),
                        ("dl_training_rows", "INTEGER NOT NULL DEFAULT 0"),
                        ("dl_trained_at", "TEXT"),
                        ("dl_error", "TEXT"),
                    ]:
                        if col_name not in training_cols:
                            c.execute(
                                f"ALTER TABLE model_training ADD COLUMN {col_name} {col_type}"
                            )
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS model_artifacts(
                            pair TEXT NOT NULL, timeframe INTEGER NOT NULL,
                            step_idx INTEGER NOT NULL, model_version TEXT NOT NULL,
                            artifact_name TEXT NOT NULL, payload BLOB NOT NULL,
                            updated_at TEXT NOT NULL,
                            PRIMARY KEY(pair, timeframe, step_idx, model_version, artifact_name)
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS predictions(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            pair TEXT NOT NULL, created_ts INTEGER NOT NULL, created_at TEXT NOT NULL,
                            ref_price REAL NOT NULL, direction TEXT, target_horizon INTEGER,
                            timeframe INTEGER DEFAULT 5, step_idx INTEGER DEFAULT 1,
                            start_ts INTEGER, end_ts INTEGER, start_price REAL,
                            prob_up REAL, prob_down REAL,
                            pat_prob REAL, stat_prob REAL, gbdt_prob REAL, dl_prob REAL, kronos_prob REAL,
                            duration_est INTEGER, duration_lo INTEGER, duration_hi INTEGER,
                            duration_confidence REAL, feature_ts INTEGER,
                            model_name TEXT, model_version TEXT,
                            status TEXT NOT NULL DEFAULT 'PENDING',
                            actual_price REAL, actual_up INTEGER, hit INTEGER,
                            settled_at TEXT, data_note TEXT
                        )
                    """)
                    cols = {r[1] for r in c.execute("PRAGMA table_info(predictions)").fetchall()}
                    for col_name, col_type in [
                        ("timeframe", "INTEGER DEFAULT 5"), ("step_idx", "INTEGER DEFAULT 1"),
                        ("start_ts", "INTEGER"), ("end_ts", "INTEGER"), ("start_price", "REAL"),
                        ("pat_prob", "REAL"), ("stat_prob", "REAL"), ("gbdt_prob", "REAL"),
                        ("dl_prob", "REAL"), ("kronos_prob", "REAL"),
                        ("feature_ts", "INTEGER"),
                    ]:
                        if col_name not in cols:
                            c.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")

                    c.execute("DROP INDEX IF EXISTS idx_pred_unique")
                    c.execute("""
                        UPDATE predictions
                        SET timeframe = CASE WHEN target_horizon IN (5,10,15,30) THEN target_horizon ELSE 5 END,
                            step_idx = 1
                        WHERE start_ts IS NULL
                    """)
                    c.execute(
                        "UPDATE predictions SET status='OBSOLETE' WHERE status='PENDING' AND COALESCE(model_version, '') != ?",
                        (MODEL_VERSION,)
                    )
                    c.execute("DROP INDEX IF EXISTS idx_pred_tf_step")
                    c.execute("""
                        DELETE FROM predictions
                        WHERE id NOT IN (SELECT MAX(id) FROM predictions GROUP BY pair, created_ts, timeframe, step_idx, model_version)
                    """)
                    c.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_tf_step
                        ON predictions(pair, created_ts, timeframe, step_idx, model_version)
                    """)
                    c.execute("""
                        CREATE INDEX IF NOT EXISTS idx_predictions_pair_status
                        ON predictions(pair, status, timeframe, step_idx)
                    """)
                    c.commit()
                    db_initialized = True
                except Exception:
                    c.close()
                    raise
    return c

def save_candles(c, pair, candles):
    now = int(time.time())
    rows, max_ts = [], None
    for x in candles or []:
        try:
            ts = int(x["from"])
            if ts + INTERVAL > now:
                continue
            rows.append((
                pair, ts, float(x["open"]), float(x["max"]),
                float(x["min"]), float(x["close"]), float(x.get("volume", 0) or 0)
            ))
            if max_ts is None or ts > max_ts:
                max_ts = ts
        except Exception:
            continue
    if not rows:
        return 0, max_ts
    before = c.total_changes
    c.executemany("INSERT OR IGNORE INTO candles(pair,ts,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)", rows)
    c.commit()
    return c.total_changes - before, max_ts

def load_initial_history(api, c, pair):
    """Load enough minute bars to build up to 1000 bars for the largest timeframe."""
    cursor = int(time.time())
    page = 0

    while True:
        count = c.execute(
            "SELECT COUNT(*) FROM candles WHERE pair=?", (pair,)
        ).fetchone()[0]
        if count >= INITIAL_HISTORY_CANDLES:
            break

        candles = call_with_timeout(
            lambda end=cursor: api.get_candles(
                pair, INTERVAL, HISTORY_CHUNK_SIZE, end
            ),
            25,
        )
        if not candles:
            add_log(f"انتهى سجل الشموع المتاح لـ {pair} بعد {count} شمعة.")
            break

        timestamps = []
        for item in candles:
            try:
                timestamps.append(int(item["from"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not timestamps:
            add_log(f"تعذر متابعة تحميل التاريخ لـ {pair}: بيانات الشموع بلا توقيت صالح.")
            break

        oldest_ts = min(timestamps)
        if oldest_ts >= cursor:
            add_log(f"توقف تحميل التاريخ لـ {pair}: المصدر لم يُرجع شموعًا أقدم.")
            break

        added, _ = save_candles(c, pair, candles)
        page += 1
        if page == 1 or page % 5 == 0:
            count = c.execute(
                "SELECT COUNT(*) FROM candles WHERE pair=?", (pair,)
            ).fetchone()[0]
            add_log(
                f"تحميل التاريخ {pair}: {count}/{INITIAL_HISTORY_CANDLES} "
                f"شمعة دقيقة | دفعة جديدة={added}"
            )
        cursor = oldest_ts - 1

    count = c.execute(
        "SELECT COUNT(*) FROM candles WHERE pair=?", (pair,)
    ).fetchone()[0]
    return count

def read_candles(c, pair):
    df = pd.read_sql_query(
        """
        SELECT ts, open, high, low, close, volume
        FROM (
            SELECT ts, open, high, low, close, volume
            FROM candles WHERE pair=? ORDER BY ts DESC LIMIT ?
        )
        ORDER BY ts
        """,
        c, params=(pair, MAX_CANDLES_IN_MEMORY)
    )
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df

def completed_minute_candles(raw: pd.DataFrame, as_of_ts=None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "time"])
    now_ts = int(time.time()) if as_of_ts is None else int(as_of_ts)
    d = raw.copy()
    d["ts"] = pd.to_numeric(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts", "open", "high", "low", "close"])
    d["ts"] = d["ts"].astype("int64")
    d = d[(d["ts"] % INTERVAL == 0) & (d["ts"] + INTERVAL <= now_ts)]
    d = d.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    if "volume" not in d:
        d["volume"] = 0.0
    d["time"] = pd.to_datetime(d["ts"], unit="s", utc=True)
    return d

def closed_timeframe_bars(raw: pd.DataFrame, tf: int, limit=MAX_TIMEFRAME_CANDLES, as_of_ts=None):
    minutes = int(tf)
    if minutes < 1:
        raise ValueError("الفريم يجب أن يكون دقيقة واحدة على الأقل")
    now_ts = int(time.time()) if as_of_ts is None else int(as_of_ts)
    minute_bars = completed_minute_candles(raw, as_of_ts=now_ts)
    if minute_bars.empty:
        return minute_bars

    width = minutes * INTERVAL
    d = minute_bars.copy()
    d["bucket_ts"] = (d["ts"] // width) * width
    bars = d.groupby("bucket_ts", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("ts", "size"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
    )
    expected_last = bars.index + (minutes - 1) * INTERVAL
    bars = bars[
        (bars["count"] == minutes)
        & (bars["first_ts"].to_numpy() == bars.index.to_numpy())
        & (bars["last_ts"].to_numpy() == expected_last.to_numpy())
        & (bars.index.to_numpy() + width <= now_ts)
    ].tail(int(limit))
    bars = bars.drop(columns=["count", "first_ts", "last_ts"])
    bars.index.name = "ts"
    bars = bars.reset_index()
    bars["time"] = pd.to_datetime(bars["ts"], unit="s", utc=True)
    return bars.reset_index(drop=True)

def detect_gaps(df):
    return 0 if (df is None or len(df) < 2) else int((df["ts"].diff().dropna() > INTERVAL * 1.5).sum())

# ===========================================================================
# 4. المؤشرات + المدة + الأنماط + الإحصاء
# ===========================================================================
def _rolling_log_slope_r2(log_vals, win):
    n = len(log_vals)
    slopes, r2s = np.full(n, np.nan), np.full(n, np.nan)
    x = np.arange(win, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(win - 1, n):
        y = log_vals[i - win + 1:i + 1]
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        slopes[i] = cov / x_var
        r2s[i] = (cov ** 2) / (x_var * ((y - y_mean) ** 2).sum() + 1e-18)
    return slopes, r2s

def indicators(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    close, high, low, op = d["close"], d["high"], d["low"], d["open"]
    log_c = np.log(close.clip(lower=1e-9))

    for n in (1, 2, 3, 5, 10, 20):
        d[f"ret{n}"] = close.pct_change(n)
    for n in (5, 10, 20, 50, 100, 200):
        d[f"sma{n}"] = close / close.rolling(n).mean() - 1
    for n in (5, 10, 20, 50, 100):
        d[f"ema{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1
    d["sma_cross_50_200"] = d["sma50"] - d["sma200"]
    d["ema_cross_20_100"] = d["ema20"] - d["ema100"]
    for n in (5, 10, 20, 50):
        d[f"gma{n}"] = close / np.exp(log_c.rolling(n).mean()) - 1
    d["gma_cross_5_20"] = d["gma5"] - d["gma20"]
    d["gma_cross_10_50"] = d["gma10"] - d["gma50"]
    d["geo_ret5"] = (close / close.shift(5).clip(lower=1e-9)) ** (1.0 / 5.0) - 1.0
    d["geo_ret15"] = (close / close.shift(15).clip(lower=1e-9)) ** (1.0 / 15.0) - 1.0

    lc_arr = log_c.to_numpy()
    s20, r2_20 = _rolling_log_slope_r2(lc_arr, 20)
    s50, _ = _rolling_log_slope_r2(lc_arr, 50)
    d["log_slope20"], d["log_r2_20"], d["log_slope50"] = s20, r2_20, s50
    d["ou_zscore"] = (log_c - log_c.rolling(30).mean()) / (log_c.rolling(30).std() + 1e-12)

    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    for n in (7, 14):
        avg_g = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
        avg_l = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
        d[f"rsi{n}"] = 100 - 100 / (1 + avg_g / (avg_l + 1e-12))

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    d["macd"] = (ema12 - ema26) / close
    d["macd_signal"] = (ema12 - ema26).ewm(span=9, adjust=False).mean() / close
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    d["bb_width"] = (upper - lower) / (mid + 1e-12)
    d["bb_position"] = (close - lower) / (upper - lower + 1e-12)

    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean() / (close + 1e-12)

    up_move, dn_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    d["plus_di"] = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / (tr14 + 1e-12)
    d["minus_di"] = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / (tr14 + 1e-12)
    dx = 100 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"] + 1e-12)
    d["adx14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    d["atr_ratio"] = d["atr14"] / (d["atr14"].rolling(50).mean() + 1e-12)
    d["bb_squeeze"] = d["bb_width"] / (d["bb_width"].rolling(50).mean() + 1e-12)
    d["dist_high20"] = (close - high.rolling(20).max()) / (close + 1e-12)
    d["dist_low20"] = (close - low.rolling(20).min()) / (close + 1e-12)
    d["dist_high60"] = (close - high.rolling(60).max()) / (close + 1e-12)
    d["dist_low60"] = (close - low.rolling(60).min()) / (close + 1e-12)

    sign = np.sign(close - op)
    d["streak"] = (sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1) * sign
    d["rsi_slope"] = d["rsi14"] - d["rsi14"].shift(3)
    d["macd_slope"] = d["macd_hist"] - d["macd_hist"].shift(2)
    d["sma_cross_5_20"] = d["sma5"] - d["sma20"]
    d["ema_cross_10_50"] = d["ema10"] - d["ema50"]

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    d["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-12)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    typical = (high + low + close) / 3
    tmean = typical.rolling(20).mean()
    mad = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["cci20"] = (typical - tmean) / (0.015 * mad + 1e-12)
    d["williams_r"] = -100 * (high14 - close) / (high14 - low14 + 1e-12)

    cr = (high - low).clip(lower=1e-12)
    body = (close - op).abs()
    d["body"] = (close - op) / (close.abs() + 1e-12)
    d["upper_shadow"] = (high - np.maximum(op, close)) / cr
    d["lower_shadow"] = (np.minimum(op, close) - low) / cr
    d["range_pct"] = cr / (close.abs() + 1e-12)
    d["doji"] = (body / cr < 0.10).astype(int)
    d["hammer"] = ((d["lower_shadow"] > 0.60) & (d["upper_shadow"] < 0.20) & (body / cr < 0.35)).astype(int)
    d["shooting_star"] = ((d["upper_shadow"] > 0.60) & (d["lower_shadow"] < 0.20) & (body / cr < 0.35)).astype(int)

    prev_bull, prev_bear = close.shift(1) > op.shift(1), close.shift(1) < op.shift(1)
    bull, bear = close > op, close < op
    d["bullish_engulfing"] = (prev_bear & bull & (op <= close.shift(1)) & (close >= op.shift(1))).astype(int)
    d["bearish_engulfing"] = (prev_bull & bear & (op >= close.shift(1)) & (close <= op.shift(1))).astype(int)

    cairo_time = d["time"].dt.tz_convert(CAIRO_TZ)
    hour = cairo_time.dt.hour
    weekday = cairo_time.dt.day_of_week

    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    d["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    ret1 = close.pct_change(1)
    vol20 = ret1.rolling(20).std()
    d["vol_z"] = (ret1 - ret1.rolling(20).mean()) / (vol20 + 1e-12)
    d["ret_vol"] = ret1 / (vol20 + 1e-12)
    d["mom10"] = close / close.shift(10) - 1
    d["mom20"] = close / close.shift(20) - 1
    d["roc10"] = close.pct_change(10)
    d["trend_strength"] = (close - close.rolling(50).mean()) / (close.rolling(20).std() + 1e-12)
    return d.replace([np.inf, -np.inf], np.nan)

def estimate_trend_duration(raw: pd.DataFrame, direction: str, atr_mult: float = 0.5, bar_minutes: int = 1):
    if raw is None or len(raw) < 200 or direction not in ("صعود", "هبوط"):
        return {"duration_est": None, "duration_lo": None, "duration_hi": None, "confidence": 0.0}
    d = indicators(raw.tail(3000))
    close = d["close"].to_numpy()
    atr = (d["atr14"] * d["close"]).to_numpy()
    n, want_up = len(close), direction == "صعود"
    durations, i = [], 50
    while i < n - 30:
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        threshold, start_price = atr_mult * a, close[i]
        window = close[i:i + 3]
        if (want_up and window.max() - start_price < threshold * 0.3) or (not want_up and start_price - window.min() < threshold * 0.3):
            i += 1
            continue
        peak, ended, dur = start_price, False, 0
        for j in range(i + 1, min(i + 180, n)):
            dur, px = j - i, close[j]
            if want_up:
                peak = max(peak, px)
                if peak - px >= threshold:
                    ended = True
                    break
            else:
                peak = min(peak, px)
                if px - peak >= threshold:
                    ended = True
                    break
        if ended and dur >= 1:
            durations.append(dur)
            i = j
        else:
            i += 1
    if len(durations) < 8:
        return {"duration_est": None, "duration_lo": None, "duration_hi": None, "confidence": 0.0}
    arr = np.array(durations, dtype=float)
    med, lo, hi = (
        int(round(np.median(arr) * bar_minutes)),
        int(round(np.percentile(arr, 25) * bar_minutes)),
        int(round(np.percentile(arr, 75) * bar_minutes)),
    )
    iqr = max(hi - lo, 1)
    conf = min(0.95, 0.25 + 0.5 * (1 - iqr / (med + iqr)) + 0.02 * min(len(arr), 40))
    return {"duration_est": med, "duration_lo": lo, "duration_hi": hi, "confidence": float(conf)}

def statistical_geo_probs(raw: pd.DataFrame, tf: int):
    sub = raw.tail(MAX_TIMEFRAME_CANDLES).reset_index(drop=True)
    closes = sub["close"].to_numpy()
    if len(closes) < 50:
        return {1: 0.5, 2: 0.5, 3: 0.5}
    log_c = np.log(np.maximum(closes, 1e-9))
    vol = float(np.std(np.diff(log_c[-40:]))) + 1e-6
    x = np.arange(min(60, len(log_c)), dtype=float)
    slope, _, r_val, _, _ = sp_stats.linregress(x, log_c[-len(x):])
    holt_steps = {1: 0.0, 2: 0.0, 3: 0.0}
    if STATSMODELS_AVAILABLE:
        try:
            if len(log_c) >= 15:
                fcast = Holt(log_c, damped_trend=True, initialization_method="estimated").fit(optimized=True).forecast(3)
                for s in STEPS:
                    holt_steps[s] = float(fcast[s - 1] - log_c[-1])
        except Exception:
            pass
    z = (log_c[-1] - np.mean(log_c[-40:])) / (np.std(log_c[-40:]) + 1e-9)
    mr_bias = -0.22 * z * vol
    out = {}
    for s in STEPS:
        drift = slope * s * (r_val ** 2) + 0.5 * holt_steps[s] + mr_bias * (0.6 ** (s - 1))
        z_pred = drift / (vol * np.sqrt(s) + 1e-6)
        out[s] = float(sp_stats.norm.cdf(np.clip(z_pred * 1.4, -2.5, 2.5)))
    return out

# ===========================================================================
# 5. التدريب والتعلم من الأخطاء
# ===========================================================================
def compute_sample_weights(df_subset, pair, tf, step):
    n = len(df_subset)
    recency = np.exp(np.linspace(-1.2, 0.0, n))
    move_pct = df_subset["future_ret"].abs().to_numpy()
    atr_pct = df_subset["atr14"].to_numpy() * np.sqrt(step)
    clarity_w = np.where(move_pct < (0.10 * np.maximum(atr_pct, 1e-6)), 0.35, 1.15)
    err_boost, mistakes = np.ones(n, dtype=float), 0
    try:
        c = db_connect()
        rows = c.execute(
            "SELECT feature_ts, hit, prob_up FROM predictions WHERE pair=? AND timeframe=? AND step_idx=? AND status='SETTLED' AND model_version=? AND feature_ts IS NOT NULL ORDER BY created_ts DESC LIMIT 1000",
            (pair, tf, step, MODEL_VERSION)
        ).fetchall()
        c.close()
        fb = {int(ts): (int(hit), float(pup or 0.5)) for ts, hit, pup in rows if hit is not None}
        if fb:
            for idx, t in enumerate(df_subset["ts"].astype(int).to_numpy()):
                if t in fb:
                    hit, pup = fb[t]
                    if hit == 0:
                        err_boost[idx] = 3.0 + 3.0 * abs(pup - 0.5)
                        mistakes += 1
                    else:
                        err_boost[idx] = 1.10
    except Exception as exc:
        add_log(f"تعذر تحميل أوزان التعلم من نتائج التوقعات: {exc}")
    w = recency * clarity_w * err_boost
    return w / np.mean(w), mistakes

def update_dl_training_status(pair, tf, step, status, error=None):
    c = db_connect()
    c.execute(
        """
        UPDATE model_training
        SET dl_status=?, dl_error=?
        WHERE pair=? AND timeframe=? AND step_idx=? AND model_version=?
        """,
        (status, error[:500] if error else None, pair, tf, step, MODEL_VERSION),
    )
    c.commit()
    c.close()

def save_model_artifact(pair, tf, step, name, payload):
    c = db_connect()
    c.execute(
        """
        INSERT OR REPLACE INTO model_artifacts(
            pair, timeframe, step_idx, model_version, artifact_name,
            payload, updated_at
        ) VALUES(?,?,?,?,?,?,?)
        """,
        (
            pair, tf, step, MODEL_VERSION, name, payload,
            datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
        ),
    )
    c.commit()
    c.close()

def save_dl_artifacts(pair, tf, step, net, scaler, test_score, training_rows):
    weights = io.BytesIO()
    torch.save(net.state_dict(), weights)
    scaler_data = io.BytesIO()
    joblib.dump(scaler, scaler_data)
    trained_at = datetime.now(CAIRO_TZ).isoformat(timespec="seconds")
    artifacts = (
        ("dl_weights", weights.getvalue()),
        ("dl_scaler", scaler_data.getvalue()),
    )
    c = db_connect()
    with c:
        for name, payload in artifacts:
            c.execute(
                """
                INSERT OR REPLACE INTO model_artifacts(
                    pair, timeframe, step_idx, model_version, artifact_name,
                    payload, updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (pair, tf, step, MODEL_VERSION, name, payload, trained_at),
            )
        c.execute(
            """
            UPDATE model_training
            SET dl_status='trained', dl_test_balanced_accuracy=?,
                dl_training_rows=?, dl_trained_at=?, dl_error=NULL
            WHERE pair=? AND timeframe=? AND step_idx=? AND model_version=?
            """,
            (
                float(test_score), int(training_rows), trained_at,
                pair, tf, step, MODEL_VERSION,
            ),
        )
    c.close()

def load_dl_artifacts(pair, tf, step):
    c = db_connect()
    rows = c.execute(
        """
        SELECT artifact_name, payload FROM model_artifacts
        WHERE pair=? AND timeframe=? AND step_idx=? AND model_version=?
          AND artifact_name IN ('dl_weights', 'dl_scaler')
        """,
        (pair, tf, step, MODEL_VERSION),
    ).fetchall()
    c.close()
    artifacts = {name: payload for name, payload in rows}
    if "dl_weights" not in artifacts or "dl_scaler" not in artifacts:
        return None
    return artifacts

def fit_sequence_network(net, x_train, y_train, sample_weights):
    positive_weight = torch.tensor(
        [max(float(len(y_train) - y_train.sum()), 1.0) / max(float(y_train.sum()), 1.0)],
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-3)
    net.train()
    for _ in range(10):
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            if len(batch) < 4:
                continue
            optimizer.zero_grad()
            loss = (
                F.binary_cross_entropy_with_logits(
                    net(x_train[batch]), y_train[batch],
                    pos_weight=positive_weight, reduction="none",
                ) * sample_weights[batch]
            ).mean()
            loss.backward()
            optimizer.step()
    net.eval()
    return net

def train_tf_step(raw, pair, tf, step):
    raw_train = closed_timeframe_bars(raw, tf, limit=MAX_CANDLES_IN_MEMORY)
    if len(raw_train) < MIN_TIMEFRAME_BARS_FOR_TRAINING:
        return f"بيانات غير كافية للفريم {tf}د: {len(raw_train)}/{MIN_TIMEFRAME_BARS_FOR_TRAINING} شمعة مكتملة."
    d = indicators(raw_train)
    s0, s1 = step - 1, step
    p_start = d["close"].shift(-s0) if s0 > 0 else d["close"]
    d["future_ret"] = d["close"].shift(-s1) / p_start - 1.0
    d["target"] = (d["future_ret"] > 0).astype(int)
    d = d.iloc[:-s1].dropna(subset=FEATURES + ["target", "future_ret"]).reset_index(drop=True)
    if len(d) < 300:
        return f"بيانات غير كافية: سُجِّل {len(d)} صفا تعلميا بعد المؤشرات، المطلوب 300."
    if d["target"].nunique() < 2:
        return "بيانات غير كافية: لم يوجد في العينة صف واحد على الأقل للصعود والهبوط."

    split = int(len(d) * 0.60)
    validation_end = int(len(d) * 0.80)
    validation_start = split + s1
    test_start = validation_end + s1
    if validation_start >= validation_end or test_start >= len(d) - 35:
        return "بيانات اختبار غير كافية."
    X, y = d[FEATURES].astype(float), d["target"].astype(int)
    w_all, mistakes = compute_sample_weights(d, pair, tf, step)
    X_tr, y_tr, w_tr = X.iloc[:split], y.iloc[:split], w_all[:split]
    X_val, y_val = X.iloc[validation_start:validation_end], y.iloc[validation_start:validation_end]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]
    if len(X_val) < 35 or len(X_te) < 35:
        return "بيانات اختبار زمني غير كافية."
    if any(part.nunique() < 2 for part in (y_tr, y_val, y_te)):
        return "توزيع الصعود والهبوط غير كافٍ في إحدى عينات الاختبار الزمني."

    def make_cands(y_ref):
        cands = {}
        cands["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=240, max_depth=12, min_samples_leaf=20,
            max_features=0.75, class_weight="balanced", random_state=42,
            n_jobs=-1
        )
        cands["RandomForest"] = RandomForestClassifier(
            n_estimators=180, max_depth=10, min_samples_leaf=25,
            max_features=0.75, class_weight="balanced", random_state=42,
            n_jobs=-1
        )
        cands["Logistic"] = LogisticRegression(
            C=0.10, max_iter=500, class_weight="balanced", random_state=42
        )
        if CATBOOST_AVAILABLE:
            cands["CatBoost"] = CatBoostClassifier(iterations=240, depth=6, learning_rate=0.04, l2_leaf_reg=5.0, loss_function="Logloss", auto_class_weights="Balanced", random_seed=42, verbose=0)
        if LIGHTGBM_AVAILABLE:
            cands["LightGBM"] = LGBMClassifier(n_estimators=220, learning_rate=0.04, num_leaves=31, min_child_samples=25, subsample=0.85, colsample_bytree=0.85, class_weight="balanced", random_state=42, verbosity=-1)
        if XGBOOST_AVAILABLE:
            pos, neg = max(int(y_ref.sum()), 1), max(int(len(y_ref) - y_ref.sum()), 1)
            cands["XGBoost"] = XGBClassifier(n_estimators=220, learning_rate=0.04, max_depth=5, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=neg / pos, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
        cands["HistGB"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=2.0, random_state=42)
        return cands

    scored = []
    validation_models = make_cands(y_tr)
    for name, clf in validation_models.items():
        try:
            add_log(f"🔬 اختيار الخوارزمية زمنيًا: {name} | {tf}د ش{step}")
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            p_all = clf.predict_proba(X_val)
            classes = list(clf.classes_)
            if 1 not in classes:
                add_log(f"⚠️ {name}: فئة الصعود غير موجودة في التدريب")
                continue

            bal = balanced_accuracy_score(y_val, clf.predict(X_val))
            brier = brier_score_loss(y_val, p_all[:, classes.index(1)])
            scored.append((bal - 0.35 * brier, name, bal, brier))
            add_log(f"📊 تحقق {name}: Balanced Accuracy={bal:.4f} | Brier={brier:.4f}")
        except Exception as exc:
            add_log(f"❌ فشل {name}: {type(exc).__name__}: {exc}")

    if not scored:
        return "فشل GBDT."
    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = scored[:3]
    development_end = validation_end
    X_dev, y_dev, w_dev = X.iloc[:development_end], y.iloc[:development_end], w_all[:development_end]
    test_baseline_brier = brier_score_loss(
        y_te, np.full(len(y_te), y_dev.mean())
    )
    development_cands = make_cands(y_dev)
    members = []
    for score, name, *_ in top3:
        est = development_cands[name]
        est.fit(X_dev, y_dev, sample_weight=w_dev)
        members.append((name, max(0.05, score), est))

    total_weight = sum(weight for _, weight, _ in members)
    test_prob_up = np.zeros(len(X_te), dtype=float)
    for _, weight, est in members:
        classes = list(est.classes_)
        if 1 in classes:
            test_prob_up += weight * est.predict_proba(X_te)[:, classes.index(1)] / total_weight
    test_bal = balanced_accuracy_score(y_te, (test_prob_up >= 0.5).astype(int))
    test_brier = brier_score_loss(y_te, test_prob_up)
    has_skill = test_bal > 0.50 and test_brier < test_baseline_brier
    final_cands = make_cands(y)
    members = []
    for score, name, *_ in top3:
        est = final_cands[name]
        est.fit(X, y, sample_weight=w_all)
        members.append((name, max(0.05, score), est))
    if not has_skill:
        add_log(
            f"ℹ️ GBDT لم يتجاوز الاختبار الزمني النهائي "
            f"(BA={test_bal:.3f} | Brier={test_brier:.4f} | "
            f"الأساس={test_baseline_brier:.4f})؛ سيشارك في التوقع مع إظهار التحفظ."
        )
    else:
        add_log(
            f"✅ اجتاز GBDT الاختبار الزمني النهائي "
            f"(BA={test_bal:.3f} | Brier={test_brier:.4f} | "
            f"الأساس={test_baseline_brier:.4f})."
        )
    brain = FusionGBDTBrain(members, usable=True, quality_passed=has_skill)
    gbdt_models[(pair, tf, step)] = brain
    serialized_brain = io.BytesIO()
    joblib.dump(brain, serialized_brain)
    save_model_artifact(pair, tf, step, "gbdt", serialized_brain.getvalue())
    joblib.dump(brain, MODEL_DIR / f"{pair}_{tf}m_c{step}_gbdt.joblib")
    db = db_connect()
    db.execute(
        """
        INSERT INTO model_training(
            pair, timeframe, step_idx, model_version, source_minute_candles,
            timeframe_candles, training_rows, quality_passed, trained_at
        ) VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(pair, timeframe, step_idx, model_version) DO UPDATE SET
            source_minute_candles=excluded.source_minute_candles,
            timeframe_candles=excluded.timeframe_candles,
            training_rows=excluded.training_rows,
            quality_passed=excluded.quality_passed,
            trained_at=excluded.trained_at
        """,
        (
            pair, tf, step, MODEL_VERSION, len(completed_minute_candles(raw)),
            len(raw_train), len(X), int(has_skill),
            datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    db.close()

    dl_bal_txt = "غير متاح" if not TORCH_AVAILABLE else "بيانات غير كافية"
    if not TORCH_AVAILABLE:
        update_dl_training_status(pair, tf, step, "unavailable")
    elif len(d) < MIN_DL_TRAINING_ROWS:
        update_dl_training_status(
            pair, tf, step, "insufficient_data",
            f"صفوف بعد المؤشرات={len(d)}، المطلوب={MIN_DL_TRAINING_ROWS}",
        )
    else:
        try:
            training_scaler = StandardScaler().fit(X_tr.to_numpy(dtype=np.float32))
            scaled_for_test = training_scaler.transform(X.to_numpy(dtype=np.float32))
            windows_for_test = np.stack([
                scaled_for_test[i - SEQUENCE_LENGTH + 1:i + 1]
                for i in range(SEQUENCE_LENGTH - 1, len(scaled_for_test))
            ])
            sequence_rows = np.arange(SEQUENCE_LENGTH - 1, len(d))
            train_mask = sequence_rows < split
            test_mask = sequence_rows >= test_start
            if train_mask.sum() < 100 or test_mask.sum() < 35:
                raise ValueError("بيانات التسلسل غير كافية للتدريب والاختبار")
            xt_tr = torch.from_numpy(windows_for_test[train_mask].astype(np.float32))
            yt_tr = torch.from_numpy(y.to_numpy(dtype=np.float32)[sequence_rows[train_mask]])
            wt_tr = torch.from_numpy(w_all[sequence_rows[train_mask]].astype(np.float32))
            xt_te = torch.from_numpy(windows_for_test[test_mask].astype(np.float32))
            y_te_seq = y.to_numpy(dtype=np.int64)[sequence_rows[test_mask]]
            torch.manual_seed(42)
            validation_net = fit_sequence_network(
                DeepCandleSequenceNet(len(FEATURES)), xt_tr, yt_tr, wt_tr
            )
            with torch.no_grad():
                p_dl = torch.sigmoid(validation_net(xt_te)).numpy()
            dl_bal = balanced_accuracy_score(y_te_seq, (p_dl >= 0.5).astype(int))
            dl_bal_txt = f"{dl_bal:.3f}"

            # Keep the chronological test untouched, then train the deployed model
            # on every valid sequence available for this pair/timeframe/target.
            deployment_scaler = StandardScaler().fit(X.to_numpy(dtype=np.float32))
            scaled_all = deployment_scaler.transform(X.to_numpy(dtype=np.float32))
            all_windows = np.stack([
                scaled_all[i - SEQUENCE_LENGTH + 1:i + 1]
                for i in range(SEQUENCE_LENGTH - 1, len(scaled_all))
            ])
            all_rows = np.arange(SEQUENCE_LENGTH - 1, len(d))
            xt_all = torch.from_numpy(all_windows.astype(np.float32))
            yt_all = torch.from_numpy(y.to_numpy(dtype=np.float32)[all_rows])
            wt_all = torch.from_numpy(w_all[all_rows].astype(np.float32))
            torch.manual_seed(42)
            deployment_net = fit_sequence_network(
                DeepCandleSequenceNet(len(FEATURES)), xt_all, yt_all, wt_all
            )
            dl_models[(pair, tf, step)] = (deployment_net, deployment_scaler, dl_bal)
            save_dl_artifacts(
                pair, tf, step, deployment_net, deployment_scaler,
                dl_bal, len(all_rows),
            )
            add_log(
                f"✅ تم تدريب GRU على كامل التاريخ: {tf}د ش{step} | "
                f"التسلسلات={len(all_rows)} | BA اختبار={dl_bal:.3f}"
            )
        except Exception as exc:
            dl_bal_txt = f"خطأ: {type(exc).__name__}"
            update_dl_training_status(
                pair, tf, step, "error", f"{type(exc).__name__}: {exc}",
            )
            add_log(f"⚠️ تعذر تدريب الشبكة العميقة ({tf}د ش{step}): {exc}")

    last_train_ts[(pair, tf, step)] = int(raw["ts"].iloc[-1])
    return (
        f"{tf}د ش{step}: GBDT={top3[0][2]:.3f} DL={dl_bal_txt} "
        f"شموع الفريم={len(raw_train)} صفوف التدريب={len(X)} "
        f"شموع الدقيقة={len(completed_minute_candles(raw))} أخطاء={mistakes}"
    )

def _load_gbdt(pair, tf, step):
    k = (pair, tf, step)
    if k in gbdt_models:
        return gbdt_models[k]
    p = MODEL_DIR / f"{pair}_{tf}m_c{step}_gbdt.joblib"
    try:
        c = db_connect()
        row = c.execute(
            """
            SELECT payload, updated_at FROM model_artifacts
            WHERE pair=? AND timeframe=? AND step_idx=? AND model_version=?
              AND artifact_name='gbdt'
            """,
            (pair, tf, step, MODEL_VERSION),
        ).fetchone()
        c.close()
        if row:
            gbdt_models[k] = joblib.load(io.BytesIO(row[0]))
            last_train_ts.setdefault(
                k, int(datetime.fromisoformat(row[1]).timestamp())
            )
            return gbdt_models[k]
    except Exception as exc:
        add_log(f"تعذر تحميل GBDT من SQLite ({tf}د ش{step}): {type(exc).__name__}: {exc}")
    if p.exists():
        try:
            gbdt_models[k] = joblib.load(p)
            serialized_brain = io.BytesIO()
            joblib.dump(gbdt_models[k], serialized_brain)
            save_model_artifact(pair, tf, step, "gbdt", serialized_brain.getvalue())
            last_train_ts.setdefault(k, int(p.stat().st_mtime))
            return gbdt_models[k]
        except Exception as exc:
            add_log(f"تعذر تحميل النموذج {p.name}: {type(exc).__name__}: {exc}")
    return None

def _load_dl(pair, tf, step):
    k = (pair, tf, step)
    if k in dl_models:
        return dl_models[k]
    if not TORCH_AVAILABLE:
        return None
    try:
        artifacts = load_dl_artifacts(pair, tf, step)
        if artifacts is not None:
            scaler = joblib.load(io.BytesIO(artifacts["dl_scaler"]))
            net = DeepCandleSequenceNet(len(FEATURES))
            state = torch.load(
                io.BytesIO(artifacts["dl_weights"]),
                map_location="cpu",
                weights_only=True,
            )
            net.load_state_dict(state)
            net.eval()
            c = db_connect()
            row = c.execute(
                """
                SELECT dl_test_balanced_accuracy FROM model_training
                WHERE pair=? AND timeframe=? AND step_idx=? AND model_version=?
                """,
                (pair, tf, step, MODEL_VERSION),
            ).fetchone()
            c.close()
            score = float(row[0]) if row and row[0] is not None else 0.5
            dl_models[k] = (net, scaler, score)
            return dl_models[k]
    except Exception as exc:
        add_log(f"تعذر تحميل نموذج GRU من SQLite ({tf}د ش{step}): {type(exc).__name__}: {exc}")

    mp, sp = MODEL_DIR / f"{pair}_{tf}m_c{step}_dl.pt", MODEL_DIR / f"{pair}_{tf}m_c{step}_dl_scaler.joblib"
    if mp.exists() and sp.exists():
        try:
            scaler = joblib.load(sp)
            net = DeepCandleSequenceNet(len(FEATURES))
            net.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
            net.eval()
            dl_models[k] = (net, scaler, 0.52)
            return dl_models[k]
        except Exception as exc:
            add_log(f"تعذر تحميل نموذج التسلسل {mp.name}: {type(exc).__name__}: {exc}")
    return None

def get_training_summary(pair):
    c = db_connect()
    rows = c.execute(
        """
        SELECT timeframe, step_idx, source_minute_candles, timeframe_candles,
               training_rows, quality_passed, dl_status,
               dl_test_balanced_accuracy, dl_training_rows, dl_error
        FROM model_training
        WHERE pair=? AND model_version=?
        """,
        (pair, MODEL_VERSION),
    ).fetchall()
    c.close()
    return {
        f"{int(tf)}_{int(step)}": {
            "minute_candles": int(minutes),
            "timeframe_candles": int(bars),
            "training_rows": int(samples),
            "quality_passed": bool(quality),
            "dl_status": dl_status,
            "dl_test_balanced_accuracy": (
                float(dl_score) if dl_score is not None else None
            ),
            "dl_training_rows": int(dl_rows),
            "dl_error": dl_error,
        }
        for tf, step, minutes, bars, samples, quality, dl_status,
            dl_score, dl_rows, dl_error in rows
    }

def get_adaptive_engine_weights(pair, tf, step):
    base = {"stat": 0.22, "gbdt": 0.42, "dl": 0.12, "kronos": 0.24}
    try:
        c = db_connect()
        rows = c.execute(
            "SELECT stat_prob, gbdt_prob, dl_prob, kronos_prob, actual_up FROM predictions WHERE pair=? AND timeframe=? AND step_idx=? AND status='SETTLED' AND model_version=? ORDER BY id DESC LIMIT 40",
            (pair, tf, step, MODEL_VERSION)
        ).fetchall()
        c.close()
        if len(rows) < 10:
            return base
        scores = {}
        for idx, name in enumerate(["stat", "gbdt", "dl", "kronos"], start=0):
            valid = [(r[idx], r[4]) for r in rows if r[idx] is not None]
            scores[name] = max(0.05, (sum(1 for p, a in valid if (1 if p >= 0.5 else 0) == a) / len(valid)) ** 2) if len(valid) >= 6 else base[name]
        tot = sum(scores.values())
        return {k: v / tot for k, v in scores.items()}
    except Exception:
        return base

def aggregate_duration(timeframe_summary, direction):
    durations = [
        item["duration"] for item in timeframe_summary.values()
        if item["direction"] == direction and item["duration"].get("duration_est") is not None
    ]
    if not durations:
        return {"duration_est": None, "duration_lo": None, "duration_hi": None, "confidence": 0.0}
    return {
        "duration_est": int(np.median([d["duration_est"] for d in durations])),
        "duration_lo": int(np.median([d["duration_lo"] for d in durations])),
        "duration_hi": int(np.median([d["duration_hi"] for d in durations])),
        "confidence": float(np.mean([d["confidence"] for d in durations])),
        "timeframes": len(durations),
    }

# ===========================================================================
# 6. التوقع + التليجرام بوقت تهدئة قابل للتحكم
# ===========================================================================
def build_forecast(raw, pair):
    raw = completed_minute_candles(raw)
    if raw is None or len(raw) < 150:
        return {"ok": False, "note": "بيانات غير كافية"}
    kronos_map = kronos_predict_all(raw, pair)
    training_summary = get_training_summary(pair)
    tf_cards, all_items, timeframe_summary = {}, [], {}
    for tf in TIMEFRAMES:
        bars = closed_timeframe_bars(raw, tf)
        d = indicators(bars)
        valid = d.dropna(subset=FEATURES).tail(MAX_TIMEFRAME_CANDLES)
        if len(valid) < SEQUENCE_LENGTH:
            tf_cards[str(tf)] = []
            continue
        row = valid.iloc[-1:]
        x_np = row[FEATURES].to_numpy(dtype=np.float32)
        stat_probs = statistical_geo_probs(bars, tf)
        candles_3 = []
        for step in STEPS:
            brain = _load_gbdt(pair, tf, step)
            gbdt_p = brain.predict_gbdt(row[FEATURES]) if brain is not None else None
            gbdt_status = "غير مدرب"
            if brain is not None:
                gbdt_status = (
                    "مدرب ومشارك واجتاز الاختبار"
                    if brain.quality_passed
                    else "مدرب ومشارك؛ أداء الاختبار ضعيف"
                )
            dl_obj = _load_dl(pair, tf, step)
            dl_p = None
            training = training_summary.get(f"{tf}_{step}")
            dl_status = "غير مدرب"
            if dl_obj is not None:
                net, scaler, _ = dl_obj
                seq = valid[FEATURES].tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
                seq = scaler.transform(seq).reshape(1, SEQUENCE_LENGTH, len(FEATURES))
                with torch.no_grad():
                    dl_p = float(torch.sigmoid(net(torch.from_numpy(seq))).item())
                if training and training["dl_test_balanced_accuracy"] is not None:
                    dl_status = (
                        "مدرب ومشارك؛ BA الاختبار "
                        f"{training['dl_test_balanced_accuracy']:.3f}"
                    )
                else:
                    dl_status = "مدرب ومشارك؛ لا توجد نتيجة اختبار محفوظة"
            elif training:
                dl_status = {
                    "pending": "بانتظار تدريب التعلم العميق",
                    "insufficient_data": "بيانات غير كافية للتعلم العميق",
                    "unavailable": "PyTorch غير متاح",
                    "error": "تعذر تدريب التعلم العميق؛ راجع السجل",
                    "trained": "النموذج محفوظ لكن تعذر تحميله",
                }.get(training["dl_status"], "غير مدرب")
            stat_p = stat_probs[step]
            kr_p = kronos_map.get((tf, step))

            w = get_adaptive_engine_weights(pair, tf, step)
            parts = [("stat", stat_p)]
            if gbdt_p is not None:
                parts.append(("gbdt", gbdt_p))
            if dl_p is not None:
                parts.append(("dl", dl_p))
            if kr_p is not None:
                parts.append(("kronos", kr_p))

            tot_w = sum(w[k] for k, _ in parts)
            final_up = sum(w[k] * p for k, p in parts) / tot_w
            conf = max(final_up, 1.0 - final_up)
            direction = "محايد" if conf < 0.55 else ("صعود" if final_up >= 0.5 else "هبوط")
            item = {
                "key": f"{tf}_{step}", "tf": tf, "step": step, "offset_min": step * tf,
                "direction": direction, "prob_up": final_up, "conf": conf,
                "stat_prob": stat_p, "gbdt_prob": gbdt_p,
                "dl_prob": dl_p, "kronos_prob": kr_p,
                "gbdt_status": gbdt_status, "dl_status": dl_status,
                "training": training,
                "feature_ts": int(bars["ts"].iloc[-1]),
                "feature_price": float(bars["close"].iloc[-1]),
            }
            candles_3.append(item)
            all_items.append(item)
        tf_cards[str(tf)] = candles_3
        first = candles_3[0]
        tf_duration = estimate_trend_duration(
            bars, first["direction"], bar_minutes=tf
        )
        timeframe_summary[str(tf)] = {
            "direction": first["direction"],
            "prob_up": first["prob_up"],
            "confidence": first["conf"],
            "duration": tf_duration,
            "candles_used": len(bars),
        }

    overview = [
        tf_cards[str(tf)][0] for tf in TIMEFRAMES
        if tf_cards.get(str(tf))
    ]
    if not overview:
        return {"ok": False, "note": "لا توجد شموع مكتملة كافية للفريمات"}
    prob_up = float(np.mean([item["prob_up"] for item in overview]))
    up_votes = sum(item["prob_up"] >= 0.5 for item in overview)
    down_votes = len(overview) - up_votes
    agreement = max(up_votes, down_votes) / len(overview)
    confidence = max(prob_up, 1.0 - prob_up)
    direction = (
        "محايد" if confidence < 0.55 or agreement < 0.75
        else ("صعود" if prob_up >= 0.5 else "هبوط")
    )
    dur = aggregate_duration(timeframe_summary, direction)
    return {
        "ok": True, "pair": pair, "ref_price": float(raw.iloc[-1]["close"]), "ref_ts": int(raw.iloc[-1]["ts"]),
        "direction": direction, "prob_up": prob_up, "prob_down": 1.0 - prob_up,
        "dir_confidence": confidence, "agreement": agreement, "duration": dur,
        "tf_cards": tf_cards, "timeframe_summary": timeframe_summary,
        "all_items": all_items,
        "candles_per_timeframe": MAX_TIMEFRAME_CANDLES,
        "training_summary": training_summary,
        "updated_at": datetime.now(CAIRO_TZ).strftime("%H:%M:%S"),
        "created_at": datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
    }

def send_telegram_message(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TG_BOT_TOKEN أو TG_CHAT_ID غير مضبوط")
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=10).json()
    if not r.get("ok"):
        raise RuntimeError(r.get("description", "فشل إرسال تليجرام"))

def test_telegram_connection():
    global tg_last_info
    try:
        send_telegram_message(
            f"✅ اختبار الاتصال ناجح (GitHub Actions)\nالعملة: {current_pair}\nالوقت: {datetime.now(CAIRO_TZ).strftime('%H:%M:%S')} بتوقيت القاهرة"
        )
        tg_last_info = f"نجح الاختبار الساعة {datetime.now(CAIRO_TZ).strftime('%H:%M:%S')} بتوقيت القاهرة"
        add_log("📩 تم إرسال رسالة اختبار التليجرام.")
    except Exception as exc:
        tg_last_info = f"فشل الاختبار: {exc}"
        add_log(f"❌ فشل اختبار التليجرام: {exc}")

def maybe_send_telegram(fc, raw):
    global tg_last_info
    if not fc or not fc.get("ok"):
        return
    if not tg_config.get("enabled"):
        tg_last_info = "معطّل (فعّل الخيار واضغط حفظ)"
        return
    selected = tg_config.get("candles") or []
    if not selected:
        tg_last_info = "لم تحدد أي شمعة من الـ 12"
        return

    pair = fc["pair"]
    now = time.time()
    if now < tg_cooldown_until.get(pair, 0):
        rem = int(tg_cooldown_until[pair] - now)
        tg_last_info = f"تهدئة ({pair}) — متبقي {max(1, rem // 60)} دقيقة"
        return

    min_conf_pct = float(tg_config.get("min_conf", 60))
    by_key = {it["key"]: it for it in fc.get("all_items", [])}
    missing = [k for k in selected if k not in by_key]
    if missing:
        tg_last_info = f"بانتظار تدريب: {', '.join(missing)}"
        return

    chosen = [by_key[k] for k in selected]
    dirs = {it["direction"] for it in chosen}
    if len(dirs) != 1:
        summary = " | ".join(f"{it['tf']}د-ش{it['step']}:{it['direction']}" for it in chosen)
        tg_last_info = f"لا يوجد توافق اتجاه بعد ({summary})"
        return
    common_dir = next(iter(dirs))
    if common_dir not in ("صعود", "هبوط"):
        tg_last_info = "الشمعات المعلمة في حالة حياد"
        return

    min_c_pct = min(round(it["conf"] * 100) for it in chosen)
    if min_c_pct < min_conf_pct:
        tg_last_info = f"الاتجاه {common_dir} لكن أقل ثقة {min_c_pct}% < المطلوب {int(min_conf_pct)}%"
        return

    dur = estimate_trend_duration(raw, common_dir)
    if dur.get("duration_est") is not None:
        est_min = max(1, int(dur["duration_est"]))
        dur_txt = f"{est_min} دقيقة (النطاق: {dur['duration_lo']}–{dur['duration_hi']} دقيقة)"
    else:
        est_min = max(it["offset_min"] for it in chosen)
        dur_txt = f"{est_min} دقيقة"

    # التحكم في مدة الانتظار قبل الإشعار التالي (0 = تلقائي، >0 = حسب اختيار المستخدم بالدقائق)
    user_cd = float(tg_config.get("cooldown_min", 0) or 0)
    wait_min = int(user_cd) if user_cd > 0 else est_min

    labels = "، ".join(f"{it['tf']}د-ش{it['step']}" for it in chosen)
    msg = (
        f"📊 إشارة تداول\n"
        f"العملة: {pair}\n"
        f"الاتجاه المتوقع: {common_dir}\n"
        f"المدة المتوقعة لحركة {common_dir}: {dur_txt}\n"
        f"توافق الشمعات ({len(chosen)}): {labels}\n"
        f"أقل نسبة ثقة: {min_c_pct}%\n"
        f"منع التكرار القادم: {wait_min} دقيقة"
    )
    try:
        send_telegram_message(msg)
        tg_cooldown_until[pair] = now + wait_min * 60
        until_str = datetime.fromtimestamp(tg_cooldown_until[pair], CAIRO_TZ).strftime("%H:%M:%S")
        tg_last_info = f"✅ أُرسل {common_dir} ({pair}) | انتظار {wait_min}د حتى {until_str}"
        add_log(f"📩 إشعار تليجرام: {pair} | {common_dir} | انتظار={wait_min}د")
    except Exception as exc:
        tg_last_info = f"❌ خطأ إرسال: {exc}"
        add_log(tg_last_info)

# ===========================================================================
# 7. حفظ وتسوية + حلقة العمل
# ===========================================================================
def save_all_3candle_predictions(fc):
    if not fc.get("ok"):
        return
    c = db_connect()
    dur = fc.get("duration") or {}
    rng = f"{dur.get('duration_lo')}-{dur.get('duration_hi')}" if dur.get("duration_est") else ""
    saved_features = {
        (int(tf), int(step)): int(ts)
        for tf, step, ts in c.execute(
            "SELECT timeframe, step_idx, MAX(feature_ts) FROM predictions WHERE pair=? AND feature_ts IS NOT NULL GROUP BY timeframe, step_idx",
            (fc["pair"],)
        ).fetchall()
    }
    for it in fc["all_items"]:
        p_up = it["prob_up"]
        tf, step = it["tf"], it["step"]
        feature_ts = int(it["feature_ts"])
        if feature_ts <= saved_features.get((tf, step), -1):
            continue
        direction = "صعود" if p_up >= 0.5 else "هبوط"
        start_ts = feature_ts + (tf - 1 + (step - 1) * tf) * INTERVAL
        end_ts = feature_ts + (tf - 1 + step * tf) * INTERVAL
        try:
            cur = c.execute("""
                INSERT OR IGNORE INTO predictions(
                    pair, created_ts, created_at, ref_price, direction,
                    target_horizon, timeframe, step_idx, start_ts, end_ts,
                    prob_up, prob_down, pat_prob, stat_prob, gbdt_prob, dl_prob, kronos_prob,
                    duration_est, duration_lo, duration_hi, duration_confidence, feature_ts,
                    model_name, model_version, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                fc["pair"], fc["ref_ts"], fc["created_at"], it["feature_price"], direction,
                tf * step, tf, step, start_ts, end_ts,
                p_up, 1.0 - p_up, None, it["stat_prob"], it["gbdt_prob"],
                it["dl_prob"], it["kronos_prob"],
                dur.get("duration_est"), dur.get("duration_lo"), dur.get("duration_hi"), dur.get("confidence"),
                it["feature_ts"],
                f"Fusion-{tf}m-C{step}", MODEL_VERSION, "PENDING"
            ))
            if cur.rowcount > 0:
                saved_features[(tf, step)] = feature_ts
                with state_lock:
                    pending_sheet_preds.append({
                        "id": cur.lastrowid, "pair": fc["pair"], "created_at": fc["created_at"],
                        "ref_price": it["feature_price"], "direction": direction,
                        "target_horizon": f"{tf}m-C{step}", "prob_up": round(p_up, 4),
                        "duration_est": dur.get("duration_est") or "", "duration_range": rng,
                        "model_name": f"Fusion-{tf}m-C{step}", "status": "PENDING"
                    })
        except Exception as exc:
            add_log(f"تعذر حفظ توقع {fc['pair']} ({tf}د ش{step}): {exc}")
            continue
    c.commit()
    c.close()

def settle_pending_predictions(raw, pair):
    if raw is None or raw.empty:
        return
    price_map = dict(zip(raw["ts"].astype(int), raw["close"].astype(float)))
    times = sorted(price_map.keys())
    c = db_connect()
    rows = c.execute(
        "SELECT id, created_ts, created_at, ref_price, direction, timeframe, step_idx, start_ts, end_ts, prob_up, duration_est, duration_lo, duration_hi FROM predictions WHERE pair=? AND status='PENDING' AND model_version=?",
        (pair, MODEL_VERSION)
    ).fetchall()
    retrain_candidates = set()
    for pid, cts, cat, ref, direction, tf, step, sts, ets, p_up, dest, dlo, dhi in rows:
        tf, step = int(tf or 5), int(step or 1)
        sts = int(sts or (cts + (step - 1) * tf * INTERVAL))
        ets = int(ets or (cts + step * tf * INTERVAL))
        p_start = next((price_map[t] for t in times if t >= sts), None)
        p_end = next((price_map[t] for t in times if t >= ets), None)
        if p_start is None or p_end is None:
            continue
        actual_up = 1 if p_end > p_start else 0
        hit = 1 if actual_up == (1 if (p_up or 0) >= 0.5 else 0) else 0
        settled_at = datetime.now(CAIRO_TZ).isoformat(timespec="seconds")
        c.execute(
            "UPDATE predictions SET status='SETTLED', start_price=?, actual_price=?, actual_up=?, hit=?, settled_at=? WHERE id=?",
            (p_start, p_end, actual_up, hit, settled_at, pid)
        )
        retrain_candidates.add((tf, step))
        with state_lock:
            pending_sheet_preds.append({
                "id": pid, "pair": pair, "created_at": cat, "ref_price": p_start,
                "direction": direction, "target_horizon": f"{tf}m-C{step}",
                "prob_up": round(p_up, 4), "duration_est": dest or "",
                "duration_range": f"{dlo}-{dhi}" if dest else "",
                "model_name": f"Fusion-{tf}m-C{step}", "status": "SETTLED",
                "actual_price": p_end, "hit": hit, "settled_at": settled_at
            })
    c.commit()
    for tf, step in retrain_candidates:
        mistakes = c.execute("SELECT COUNT(*) FROM predictions WHERE pair=? AND timeframe=? AND step_idx=? AND status='SETTLED' AND hit=0 AND model_version=?", (pair, tf, step, MODEL_VERSION)).fetchone()[0]
        last12 = c.execute("SELECT hit FROM predictions WHERE pair=? AND timeframe=? AND step_idx=? AND status='SETTLED' AND model_version=? ORDER BY id DESC LIMIT 12", (pair, tf, step, MODEL_VERSION)).fetchall()
        recent_acc = (sum(x[0] for x in last12) / len(last12)) if len(last12) >= 10 else 1.0
        prev_m = last_mistake_retrain.get((pair, tf, step), 0)
        if time.time() - last_retrain_ts.get((pair, tf, step), 0) > 600 and (mistakes - prev_m >= 6 or recent_acc < 0.50):
            last_mistake_retrain[(pair, tf, step)] = mistakes
            last_retrain_ts[(pair, tf, step)] = time.time()
            add_log(f"🔄 تصحيح ذاتي ({pair} | {tf}د ش{step}): أخطاء={mistakes}")
            k = (pair, tf, step)
            with training_lock:
                if k not in training_jobs:
                    training_jobs.add(k)
                    train_queue.put((raw.copy(), pair, tf, step))
    c.close()

def prediction_stats(pair):
    c = db_connect()
    row = c.execute("SELECT COUNT(*), SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END), SUM(CASE WHEN hit=0 THEN 1 ELSE 0 END) FROM predictions WHERE pair=? AND status='SETTLED'", (pair,)).fetchone()
    pending = c.execute("SELECT COUNT(*) FROM predictions WHERE pair=? AND status='PENDING' AND model_version=?", (pair, MODEL_VERSION)).fetchone()[0]
    by_tf_step = {}
    for tf in TIMEFRAMES:
        for step in STEPS:
            r = c.execute("SELECT COUNT(*), SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END) FROM predictions WHERE pair=? AND timeframe=? AND step_idx=? AND status='SETTLED'", (pair, tf, step)).fetchone()
            sc, hc = int(r[0] or 0), int(r[1] or 0)
            by_tf_step[f"{tf}_{step}"] = {"settled": sc, "hits": hc, "hit_rate": (100.0 * hc / sc) if sc else None}
    by_pair = []
    for p, cnt, hits in c.execute("SELECT pair, COUNT(*), SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END) FROM predictions WHERE status='SETTLED' GROUP BY pair ORDER BY COUNT(*) DESC").fetchall():
        if p not in FOREX_PAIRS:
            continue
        cnt, hits = int(cnt or 0), int(hits or 0)
        by_pair.append({"pair": p, "settled": cnt, "hits": hits, "hit_rate": (100.0 * hits / cnt) if cnt else None})
    c.close()
    settled, hits, mistakes = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
    return {
        "settled": settled, "hits": hits, "mistakes": mistakes,
        "hit_rate": (100.0 * hits / settled) if settled else None,
        "pending": int(pending or 0), "by_tf_step": by_tf_step, "by_pair": by_pair,
    }


def train_loop():
    global latest_forecast, active_training

    while not stop_event.is_set():
        try:
            job = train_queue.get(timeout=1)
        except queue.Empty:
            continue

        if job is None:
            train_queue.task_done()
            break

        raw, pair, tf, step = job
        k = (pair, tf, step)

        active_training = k

        try:
            add_log(
                f"🧠 بدء التدريب: {pair} | "
                f"{tf} دقيقة | الشمعة {step} | "
                f"البيانات={len(raw)}"
            )

            info = train_tf_step(raw, pair, tf, step)

            add_log(f"✅ نتيجة التدريب: {info}")


        except Exception as exc:
            add_log(
                f"❌ خطأ تدريب ({tf}د ش{step}): "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            with training_lock:
                training_jobs.discard(k)

            active_training = None
            train_queue.task_done()


def enqueue_training(raw, pair, force=False):
    if len(raw) < MIN_CANDLES_FOR_TRAINING:
        add_log(f"انتظار البيانات: {len(raw)}/{MIN_CANDLES_FOR_TRAINING} شمعة")
        return

    latest_ts = int(raw["ts"].iloc[-1])
    added = 0
    training_summary = get_training_summary(pair) if TORCH_AVAILABLE else {}

    for tf in TIMEFRAMES:
        tf_bar_count = len(closed_timeframe_bars(
            raw, tf, limit=MAX_CANDLES_IN_MEMORY
        ))
        if tf_bar_count < MIN_TIMEFRAME_BARS_FOR_TRAINING:
            continue
        for step in STEPS:
            k = (pair, tf, step)
            training_record = training_summary.get(f"{tf}_{step}")
            dl_needs_training = (
                TORCH_AVAILABLE
                and tf_bar_count >= MIN_DL_TRAINING_ROWS
                and (
                    not training_record
                    or training_record["dl_status"] != "trained"
                    or _load_dl(pair, tf, step) is None
                )
            )

            with training_lock:
                if k in training_jobs:
                    continue

                model_missing = _load_gbdt(pair, tf, step) is None
                previous_ts = last_train_ts.get(k)
                if previous_ts is None and not model_missing:
                    model_path = MODEL_DIR / f"{pair}_{tf}m_c{step}_gbdt.joblib"
                    previous_ts = int(model_path.stat().st_mtime) if model_path.exists() else 0
                    last_train_ts[k] = previous_ts
                new_data = previous_ts is None or latest_ts - previous_ts >= RETRAIN_NEW_CANDLES * INTERVAL

                if not (force or new_data or dl_needs_training):
                    continue

                training_jobs.add(k)

            train_queue.put((raw.copy(), pair, tf, step))
            added += 1

    if added:
        add_log(
            f"📚 أُضيفت {added} مهمة تدريب | "
            f"المهام المنتظرة={train_queue.qsize()}"
        )
    else:
        add_log("لا توجد مهام تدريب جديدة مطلوبة")


def push_to_cloud(raw, pair, max_ts, gaps, status_text):
    with state_lock:
        preds = list(pending_sheet_preds)
        pending_sheet_preds.clear()
        logs = list(logs_buffer)
    tail = [{"ts": int(r.ts), "close": float(r.close)} for r in raw.tail(50).itertuples()] if raw is not None and not raw.empty else []
    fc_send = {k: v for k, v in latest_forecast.items() if k != "all_items"} if latest_forecast and latest_forecast.get("ok") else latest_forecast
    cd_left = max(0, int(tg_cooldown_until.get(pair, 0) - time.time()))
    payload = {
        "action": "push_state", "token": SECRET_TOKEN,
        "state": {
            "ok": True, "engine_status": status_text, "pair": pair,
            "candle_count": int(len(raw)) if raw is not None else 0,
            "last_candle_str": datetime.fromtimestamp(max_ts, CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S") if max_ts else "—",
            "gaps": gaps, "forecast": fc_send, "stats": prediction_stats(pair),
            "tg_status": {"info": tg_last_info, "cooldown_sec": cd_left},
            "candles_tail": tail, "logs": logs,
        },
        "predictions": preds,
    }
    try:
        response = requests.post(APPS_SCRIPT_URL, json=payload, timeout=12)
        response.raise_for_status()
        result = response.json()
        if result.get("ok") is False:
            raise RuntimeError(result.get("error", "رفض Apps Script تحديث الحالة"))
    except Exception as exc:
        add_log(f"تعذر الإرسال للسحابة: {exc}")

def poll_cloud_command():
    global tg_config
    try:
        r = requests.get(
            APPS_SCRIPT_URL,
            params={"action": "poll_command", "token": SECRET_TOKEN, "consume": "1"},
            timeout=12
        ).json()
        new_cfg = r.get("tg_config")
        if isinstance(new_cfg, dict) and new_cfg != tg_config:
            tg_config = new_cfg
            add_log(f"⚙️ تحديث تليجرام: مفعل={tg_config.get('enabled')} | تهدئة={tg_config.get('cooldown_min')}د")
        return r.get("command")
    except Exception as exc:
        add_log(f"تعذر استطلاع أمر السحابة: {exc}")
        return None

def stop_workflow_continuation():
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write("continue=false\n")

if __name__ == "__main__":
    missing_secrets = [
        name for name, value in {
            "IQ_EMAIL": IQ_EMAIL,
            "IQ_PASSWORD": IQ_PASSWORD,
            "APPS_SCRIPT_URL": APPS_SCRIPT_URL,
            "SECRET_TOKEN": SECRET_TOKEN,
        }.items() if not value
    ]
    if missing_secrets:
        raise SystemExit(f"إعدادات GitHub Secrets المطلوبة غير موجودة: {', '.join(missing_secrets)}")

    if DEFAULT_PAIR not in FOREX_PAIRS:
        add_log(f"الزوج الافتراضي {DEFAULT_PAIR} غير مدعوم؛ سيستخدم EURUSD.")

    start_ts = time.time()
    threading.Thread(target=train_loop, daemon=True).start()
    c = db_connect()
    api = None
    last_forecast_key = None
    last_settlement_key = None

    # إرسال فوري للويب منذ الثانية الأولى حتى ترى الحالة الحية
    add_log("🚀 بدء تشغيل المحرك على GitHub Actions...")
    push_to_cloud(read_candles(c, current_pair), current_pair, None, 0, "جاري تحميل Kronos والاتصال...")

    # تحميل Kronos في الخلفية حتى لا يعطّل الاتصال
    threading.Thread(target=init_kronos, daemon=True).start()

    try:
        while not stop_event.is_set():
            if MAX_RUNTIME_SEC > 0 and time.time() - start_ts >= MAX_RUNTIME_SEC:
                add_log("انتهاء مدة الجلسة المجدولة.")
                break

            cmd = poll_cloud_command()
            if cmd:
                if cmd.get("stop"):
                    add_log("أمر إيقاف من الويب.")
                    stop_workflow_continuation()
                    break
                if cmd.get("test_telegram"):
                    test_telegram_connection()
                requested_pair = cmd.get("pair")
                if requested_pair:
                    if not isinstance(requested_pair, str) or requested_pair.strip().upper() not in FOREX_PAIRS:
                        add_log(f"رُفض تغيير العملة إلى زوج غير مدعوم: {requested_pair}")
                    elif requested_pair.strip().upper() != current_pair:
                        current_pair = requested_pair.strip().upper()
                        add_log(f"تغيير الزوج إلى: {current_pair}")
                if cmd.get("train_now"):
                    raw = read_candles(c, current_pair)
                    enqueue_training(raw, current_pair, force=True)

            if datetime.now(CAIRO_TZ).weekday() in (5, 6):
                market_closed_note = "سوق الفوركس مغلق السبت والأحد بتوقيت القاهرة"
                latest_forecast = {"ok": False, "note": market_closed_note}
                push_to_cloud(
                    read_candles(c, current_pair), current_pair, None, 0,
                    market_closed_note
                )
                stop_event.wait(60)
                continue

            try:
                if api is None or not api.check_connect():
                    add_log("جاري الاتصال بـ IQ Option...")
                    push_to_cloud(read_candles(c, current_pair), current_pair, None, 0, "جاري الاتصال بـ IQ Option...")
                    def _connect():
                        a = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
                        r = a.connect()
                        ok = r[0] if isinstance(r, tuple) else bool(r)
                        if not ok or not a.check_connect():
                            raise RuntimeError(f"فشل دخول IQ Option: {r}")
                        return a
                    api = call_with_timeout(_connect, 25)
                    add_log("✅ اتصال ناجح بـ IQ Option.")

                    load_initial_history(api, c, current_pair)
                    raw = read_candles(c, current_pair)
                    enqueue_training(raw, current_pair)

                candles = call_with_timeout(lambda: api.get_candles(current_pair, INTERVAL, 60, int(time.time())), 20)
                n, max_ts = save_candles(c, current_pair, candles)
                raw = read_candles(c, current_pair)
                gaps = detect_gaps(raw)

                data_key = (current_pair, int(raw.iloc[-1]["ts"])) if not raw.empty else None
                if data_key is not None and (n > 0 or data_key != last_settlement_key):
                    settle_pending_predictions(raw, current_pair)
                    last_settlement_key = data_key
                if len(raw) >= 150:
                    forecast_key = data_key
                    if n > 0 or forecast_key != last_forecast_key:
                        latest_forecast = build_forecast(raw, current_pair)
                        last_forecast_key = forecast_key
                        if latest_forecast.get("ok"):
                            if n > 0:
                                save_all_3candle_predictions(latest_forecast)
                            maybe_send_telegram(latest_forecast, raw)
                if n > 0:
                    add_log(f"شموع جديدة={n} | الإجمالي={len(raw)}")
                    enqueue_training(raw, current_pair)

                push_to_cloud(
                    raw, current_pair, max_ts, gaps,
                    "يعمل ضمن جلسة GitHub Actions المجدولة"
                )
            except Exception as exc:
                add_log(f"⚠️ مشكلة اتصال مؤقتة: {exc}")
                api = None
                push_to_cloud(read_candles(c, current_pair), current_pair, None, 0, f"إعادة محاولة: {exc}")

            stop_event.wait(15)
    finally:
        train_queue.put(None)
        train_queue.join()
        stop_event.set()
        final_raw = read_candles(c, current_pair)
        if len(final_raw) >= 150:
            try:
                latest_forecast = build_forecast(final_raw, current_pair)
            except Exception as exc:
                add_log(
                    f"تعذر تحديث التوقع بعد اكتمال التدريب: "
                    f"{type(exc).__name__}: {exc}"
                )
        push_to_cloud(
            final_raw, current_pair, None, 0,
            "تم حفظ التدريب والبيانات؛ بانتظار التشغيل التالي"
        )
        c.close()
