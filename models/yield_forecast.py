"""
Advanced yield forecasting utilities combining modern time-series modelling
with gradient boosted decision trees and phenology-aware fallbacks.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:  # NeuralProphet captures recent advances in temporal forecasting
    from neuralprophet import NeuralProphet

    HAS_NEURALPROPHET = True
except Exception:  # pragma: no cover - optional dependency
    HAS_NEURALPROPHET = False

try:  # Fallback to Prophet if NeuralProphet is unavailable
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None

logger = logging.getLogger(__name__)

# Baseline yields (t/ha) taken from FAOSTAT 2023 world averages
CROP_BASELINE_YIELD = {
    "Wheat": 3.5,
    "Corn": 6.2,
    "Maize": 6.2,
    "Soybean": 2.8,
    "Barley": 3.2,
    "Canola": 2.3,
    "Rice": 4.7,
    "Oats": 2.4,
    "Rye": 2.7,
    "Sunflower": 2.0,
    "Other": 3.0,
}

# Phenology-driven response weights inspired by recent remote-sensing literature
# (e.g. You et al. 2023, Wageningen UR yield forecasting benchmarks).
CROP_RESPONSE_WEIGHTS = {
    "Wheat": {"ndvi_peak": 1.55, "ndvi_integral": 0.018, "trend": 0.65, "precip": 0.003},
    "Corn": {"ndvi_peak": 1.75, "ndvi_integral": 0.021, "trend": 0.70, "precip": 0.004},
    "Soybean": {"ndvi_peak": 1.60, "ndvi_integral": 0.016, "trend": 0.60, "precip": 0.003},
    "Barley": {"ndvi_peak": 1.40, "ndvi_integral": 0.017, "trend": 0.55, "precip": 0.002},
    "Canola": {"ndvi_peak": 1.35, "ndvi_integral": 0.014, "trend": 0.50, "precip": 0.002},
    "Rice": {"ndvi_peak": 1.80, "ndvi_integral": 0.024, "trend": 0.75, "precip": 0.005},
    "Oats": {"ndvi_peak": 1.30, "ndvi_integral": 0.015, "trend": 0.50, "precip": 0.002},
    "Rye": {"ndvi_peak": 1.25, "ndvi_integral": 0.014, "trend": 0.45, "precip": 0.002},
    "Sunflower": {"ndvi_peak": 1.45, "ndvi_integral": 0.018, "trend": 0.55, "precip": 0.003},
    "Other": {"ndvi_peak": 1.45, "ndvi_integral": 0.017, "trend": 0.55, "precip": 0.003},
}


class YieldForecastModel:
    """Forecast crop yields from NDVI and weather time series."""

    def __init__(self) -> None:
        self.temporal_model: Any = None
        self.temporal_backend: Optional[str] = None
        self.ndvi_forecast: Optional[pd.DataFrame] = None
        self.lgbm_model: Optional[lgb.LGBMRegressor] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.residual_std: Optional[float] = None
        self.models_dir = Path("./models/saved")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.use_fallback = False
        self.last_raw_frame: Optional[pd.DataFrame] = None
        self.engineered_frame: Optional[pd.DataFrame] = None
        self.minimum_target_points = 45  # ~6 weeks of daily data

    # ------------------------------------------------------------------ #
    # Data preparation and feature engineering
    # ------------------------------------------------------------------ #
    def prepare_training_data(
        self,
        ndvi_time_series: Dict[str, float],
        weather_data: Optional[Dict[str, Dict[str, float]]] = None,
        historical_yields: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        if not ndvi_time_series:
            return pd.DataFrame()

        ndvi_df = (
            pd.DataFrame([(date, value) for date, value in ndvi_time_series.items()], columns=["date", "ndvi"])
            .dropna(subset=["ndvi"])
        )
        ndvi_df["date"] = pd.to_datetime(ndvi_df["date"])
        ndvi_df = ndvi_df.drop_duplicates("date").sort_values("date")

        # Promote NDVI to a daily signal for better alignment with weather
        daily_ndvi = (
            ndvi_df.set_index("date")
            .resample("D")
            .interpolate(method="linear")
            .ffill()
            .bfill()
            .reset_index()
        )

        weather_data = weather_data or {}
        weather_rows = []
        for date, payload in weather_data.items():
            row = {"date": pd.to_datetime(date)}
            row.update(payload or {})
            weather_rows.append(row)
        weather_df = pd.DataFrame(weather_rows).sort_values("date") if weather_rows else pd.DataFrame(columns=["date"])

        if weather_df.empty:
            merged_df = daily_ndvi.copy()
        else:
            merged_df = pd.merge(daily_ndvi, weather_df, on="date", how="outer")

        merged_df = merged_df.sort_values("date").ffill().bfill()
        merged_df["year"] = merged_df["date"].dt.year

        if historical_yields:
            yield_df = pd.DataFrame(
                [(int(year), value) for year, value in historical_yields.items()],
                columns=["year", "yield"],
            )
            merged_df = pd.merge(merged_df, yield_df, on="year", how="left")
            merged_df["yield"] = merged_df.groupby("year")["yield"].transform(lambda s: s.ffill().bfill())

        self.last_raw_frame = merged_df.copy()
        engineered = self._engineer_features(merged_df)
        self.engineered_frame = engineered.copy()
        return engineered

    def _engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy().sort_values("date")
        if "ndvi" not in df.columns:
            df["ndvi"] = np.nan

        for lag in [1, 3, 7, 14, 30]:
            df[f"ndvi_lag_{lag}"] = df["ndvi"].shift(lag)
            df[f"ndvi_delta_{lag}"] = df["ndvi"] - df[f"ndvi_lag_{lag}"]

        df["ndvi_rolling_mean_7"] = df["ndvi"].rolling(7, min_periods=1).mean()
        df["ndvi_rolling_mean_30"] = df["ndvi"].rolling(30, min_periods=1).mean()
        df["ndvi_rolling_std_30"] = df["ndvi"].rolling(30, min_periods=1).std()
        df["ndvi_integral_45"] = df["ndvi"].rolling(45, min_periods=1).sum()
        df["ndvi_peak_45"] = df["ndvi"].rolling(45, min_periods=1).max()

        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["month"] = df["date"].dt.month

        weather_cols = {"temp_max", "temp_min", "precip", "evapotrans"}
        if {"temp_max", "temp_min"}.issubset(df.columns):
            df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
            df["gdd"] = (df["temp_mean"] - 10).clip(lower=0)
            df["gdd_cum"] = df["gdd"].cumsum()
            df["heat_stress"] = (df["temp_max"] - 32).clip(lower=0)
            df["heat_stress_cum"] = df["heat_stress"].cumsum()

        if "precip" in df.columns:
            df["precip_rolling_15"] = df["precip"].rolling(15, min_periods=1).sum()
            df["precip_rolling_30"] = df["precip"].rolling(30, min_periods=1).sum()
            df["precip_anomaly"] = df["precip"] - df["precip"].rolling(30, min_periods=1).mean()

        if "evapotrans" in df.columns:
            df["water_balance_30"] = (df["precip_rolling_30"] - df["evapotrans"].rolling(30, min_periods=1).sum()).fillna(0)

        df["ndvi_anomaly"] = df["ndvi"] - df.groupby("month")["ndvi"].transform("median")

        df = df.ffill().bfill()
        max_gap = 30
        df = df.iloc[max_gap:].reset_index(drop=True) if len(df) > max_gap else df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Training routines
    # ------------------------------------------------------------------ #
    def train_models(self, forecast_periods: int = 120) -> bool:
        if self.engineered_frame is None or self.engineered_frame.empty:
            self.use_fallback = True
            return False

        temporal_source = self.last_raw_frame[["date", "ndvi"]].dropna() if self.last_raw_frame is not None else pd.DataFrame()
        if len(temporal_source) >= 30:
            self._train_temporal_model(temporal_source, forecast_periods)
        else:
            self.temporal_model = None
            self.ndvi_forecast = None
            self.temporal_backend = None

        has_yield = "yield" in self.engineered_frame.columns and self.engineered_frame["yield"].notna().sum() >= self.minimum_target_points
        if has_yield:
            return self._train_lightgbm_model(self.engineered_frame)

        self.use_fallback = True
        return False

    def _train_temporal_model(self, time_series_df: pd.DataFrame, forecast_periods: int) -> None:
        prophet_df = time_series_df.rename(columns={"date": "ds", "ndvi": "y"})

        if HAS_NEURALPROPHET:
            model = NeuralProphet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                learning_rate=0.01,
                epochs=200,
                batch_size=64,
            )
            model.fit(prophet_df, freq="D", progress="none")
            future = model.make_future_dataframe(prophet_df, periods=forecast_periods)
            forecast = model.predict(future)
            mean_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
            lower_col = "yhat1_lower" if "yhat1_lower" in forecast.columns else None
            upper_col = "yhat1_upper" if "yhat1_upper" in forecast.columns else None

            ndvi_forecast = forecast[["ds", mean_col]].rename(columns={"ds": "date", mean_col: "mean"})
            if lower_col and upper_col:
                ndvi_forecast["lower"] = forecast[lower_col]
                ndvi_forecast["upper"] = forecast[upper_col]
            else:
                ndvi_forecast["lower"] = ndvi_forecast["mean"] * 0.9
                ndvi_forecast["upper"] = ndvi_forecast["mean"] * 1.1

            self.temporal_model = model
            self.temporal_backend = "neuralprophet"
            self.ndvi_forecast = ndvi_forecast
            return

        if Prophet is None:
            logger.warning("No temporal forecasting backend available; falling back to persistence NDVI.")
            self.temporal_model = None
            self.temporal_backend = None
            self.ndvi_forecast = None
            return

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
        )
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        ndvi_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"ds": "date", "yhat": "mean", "yhat_lower": "lower", "yhat_upper": "upper"}
        )

        self.temporal_model = model
        self.temporal_backend = "prophet"
        self.ndvi_forecast = ndvi_forecast

    def _train_lightgbm_model(self, features_df: pd.DataFrame, target_column: str = "yield") -> bool:
        dataset = features_df.dropna(subset=[target_column]).copy()
        if len(dataset) < self.minimum_target_points:
            self.use_fallback = True
            return False

        X = dataset.drop(columns=[target_column]).copy()
        if "date" in X.columns:
            X["date_ordinal"] = pd.to_datetime(dataset["date"]).map(dt.datetime.toordinal)
            X = X.drop(columns=["date"])

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        y = dataset[target_column].astype(float)

        max_splits = min(5, max(2, len(X) // 30))
        if max_splits < 2:
            self.use_fallback = True
            return False

        tscv = TimeSeriesSplit(n_splits=max_splits)
        param_space = {
            "num_leaves": [31, 63, 127, 255],
            "learning_rate": [0.005, 0.01, 0.02, 0.05],
            "feature_fraction": [0.7, 0.8, 0.9, 1.0],
            "bagging_fraction": [0.6, 0.8, 1.0],
            "bagging_freq": [0, 1, 5],
            "min_child_samples": [10, 20, 40],
            "lambda_l1": [0.0, 0.1, 0.3],
            "lambda_l2": [0.0, 0.1, 0.3],
        }

        base_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=20,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        best_model.fit(X, y)

        preds, actuals = [], []
        for train_idx, test_idx in tscv.split(X, y):
            fold_model = lgb.LGBMRegressor(**best_model.get_params())
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_pred = fold_model.predict(X.iloc[test_idx])
            preds.append(fold_pred)
            actuals.append(y.iloc[test_idx].values)

        if preds:
            preds_array = np.concatenate(preds)
            actual_array = np.concatenate(actuals)
            self.metrics = {
                "mae": float(mean_absolute_error(actual_array, preds_array)),
                "rmse": float(np.sqrt(mean_squared_error(actual_array, preds_array))),
                "r2": float(r2_score(actual_array, preds_array)),
            }
            residuals = actual_array - preds_array
            self.residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else None

        booster = best_model.booster_
        importance = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        self.feature_importance = (
            pd.DataFrame({"Feature": names, "Importance": importance})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        self.lgbm_model = best_model
        self.feature_columns = numeric_cols
        self.use_fallback = False
        return True

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #
    def forecast_from_series(
        self,
        ndvi_time_series: Dict[str, float],
        crop_type: str,
        forecast_dates: List[str],
        weather_data: Optional[Dict[str, Dict[str, float]]] = None,
        historical_yields: Optional[Dict[str, float]] = None,
        forecast_periods: int = 150,
    ) -> Dict[str, Any]:
        engineered = self.prepare_training_data(ndvi_time_series, weather_data, historical_yields)
        if engineered.empty:
            return {}

        self.train_models(forecast_periods=forecast_periods)
        crop_key = self._resolve_crop_name(crop_type)

        predictions, intervals = self._forecast_yield_series(forecast_dates, crop_key)
        forecasts = {
            crop_key: {date: round(value, 2) for date, value in predictions.items()}
        }

        metadata: Dict[str, Any] = {
            "crop": crop_key,
            "model_backend": "phenology_fallback" if self.use_fallback else "lightgbm+temporal",
            "temporal_model": self.temporal_backend,
            "intervals": {d: {"lower": round(low, 2), "upper": round(high, 2)} for d, (low, high) in intervals.items()},
        }
        if self.metrics:
            metadata["metrics"] = self.metrics
        if self.feature_importance is not None:
            metadata["feature_importance"] = self.feature_importance.to_dict(orient="records")

        return {"forecasts": forecasts, "metadata": metadata}

    def _forecast_yield_series(
        self,
        forecast_dates: List[str],
        crop_key: str,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        if self.use_fallback or self.lgbm_model is None or self.feature_columns == []:
            return self._fallback_forecast(forecast_dates, crop_key)

        if self.last_raw_frame is None:
            return self._fallback_forecast(forecast_dates, crop_key)

        raw_df = self.last_raw_frame.copy().sort_values("date")
        predictions: Dict[str, float] = {}
        intervals: Dict[str, Tuple[float, float]] = {}

        for horizon, date_str in enumerate(forecast_dates):
            date_ts = pd.to_datetime(date_str)
            ndvi_value = self._forecasted_ndvi_value(date_ts)

            if raw_df["date"].iloc[-1] >= date_ts:
                working_df = raw_df.copy()
            else:
                template_row = raw_df.iloc[-1].copy()
                template_row["date"] = date_ts
                template_row["ndvi"] = ndvi_value
                for col in template_row.index:
                    if col not in {"date", "ndvi", "year"} and pd.isna(template_row[col]):
                        template_row[col] = raw_df[col].ffill().iloc[-1] if col in raw_df else np.nan
                raw_df = pd.concat([raw_df, pd.DataFrame([template_row])], ignore_index=True)
                raw_df["year"] = raw_df["date"].dt.year
                working_df = raw_df.copy()

            engineered = self._engineer_features(working_df)
            latest_row = engineered.iloc[-1].copy()
            feature_row = latest_row[self.feature_columns].to_frame().T
            pred = float(self.lgbm_model.predict(feature_row)[0])

            interval_half = self._prediction_interval_width(pred, horizon)
            predictions[date_str] = pred
            intervals[date_str] = (max(pred - interval_half, 0.0), max(pred + interval_half, 0.0))

        return predictions, intervals

    def _forecasted_ndvi_value(self, target_date: pd.Timestamp) -> float:
        if self.ndvi_forecast is None or self.ndvi_forecast.empty:
            if self.last_raw_frame is not None and not self.last_raw_frame.empty:
                return float(self.last_raw_frame["ndvi"].ffill().iloc[-1])
            return 0.5

        ndvi_df = self.ndvi_forecast.copy()
        ndvi_df["date"] = pd.to_datetime(ndvi_df["date"])
        match = ndvi_df[ndvi_df["date"] == target_date]
        if not match.empty:
            return float(match.iloc[0]["mean"])

        prior = ndvi_df[ndvi_df["date"] < target_date]
        if not prior.empty:
            return float(prior.iloc[-1]["mean"])

        return float(ndvi_df["mean"].iloc[-1])

    def _prediction_interval_width(self, prediction: float, horizon_index: int) -> float:
        base = 1.64 * self.residual_std if self.residual_std else max(prediction * 0.15, 0.4)
        decay = 1 + (horizon_index * 0.15)
        return base * decay

    def _fallback_forecast(
        self,
        forecast_dates: List[str],
        crop_key: str,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        if self.last_raw_frame is None or self.last_raw_frame.empty:
            return {}, {}

        df = self.last_raw_frame.copy().sort_values("date")
        ndvi_series = df["ndvi"].dropna()
        if ndvi_series.empty:
            return {}, {}

        window = min(45, len(ndvi_series))
        peak_ndvi = float(ndvi_series.rolling(window, min_periods=1).max().iloc[-1])
        ndvi_integral = float(ndvi_series.iloc[-window:].sum())
        ndvi_trend = float((ndvi_series.iloc[-1] - ndvi_series.iloc[-window]) / max(window, 1))

        precip_sum = 0.0
        if "precip" in df.columns:
            precip_sum = float(df["precip"].rolling(window, min_periods=1).sum().iloc[-1])

        coeffs = CROP_RESPONSE_WEIGHTS.get(crop_key, CROP_RESPONSE_WEIGHTS["Other"])
        baseline = CROP_BASELINE_YIELD.get(crop_key, CROP_BASELINE_YIELD["Other"])

        quality_factor = (
            0.45
            + coeffs["ndvi_peak"] * (peak_ndvi - 0.5)
            + coeffs["ndvi_integral"] * (ndvi_integral / window)
            + coeffs["trend"] * ndvi_trend
            + coeffs["precip"] * (precip_sum / max(window, 1))
        )
        quality_factor = max(0.2, min(1.8, quality_factor))

        predictions: Dict[str, float] = {}
        intervals: Dict[str, Tuple[float, float]] = {}

        for idx, date in enumerate(forecast_dates):
            horizon_decay = max(0.7, 1 - (idx * 0.07))
            value = baseline * quality_factor * horizon_decay
            interval_width = max(value * 0.2, 0.6 + idx * 0.25)
            predictions[date] = value
            intervals[date] = (max(value - interval_width, 0.0), max(value + interval_width, 0.0))

        return predictions, intervals

    # ------------------------------------------------------------------ #
    # Persistence utilities retained for compatibility
    # ------------------------------------------------------------------ #
    def save_model(self, filename: str = "yield_forecast_model") -> str:
        if self.lgbm_model is None and self.temporal_model is None:
            logger.error("No trained models to save")
            return ""

        model_dir = self.models_dir / filename
        model_dir.mkdir(exist_ok=True, parents=True)

        if self.lgbm_model is not None:
            lgbm_path = model_dir / "lightgbm_model.txt"
            self.lgbm_model.booster_.save_model(str(lgbm_path))
            if self.feature_importance is not None:
                self.feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)
            if self.metrics:
                with open(model_dir / "metrics.json", "w") as fh:
                    json.dump(self.metrics, fh, indent=2)

        if self.temporal_model is not None and self.temporal_backend == "prophet":
            prophet_path = model_dir / "prophet_model.json"
            with open(prophet_path, "w") as fh:
                json.dump(self.temporal_model.to_json(), fh, indent=2)

        return str(model_dir)

    def load_model(self, model_dir: str) -> bool:
        model_path = Path(model_dir)
        try:
            lgbm_path = model_path / "lightgbm_model.txt"
            if lgbm_path.exists():
                booster = lgb.Booster(model_file=str(lgbm_path))
                self.lgbm_model = lgb.LGBMRegressor()
                self.lgbm_model._Booster = booster
                self.feature_columns = booster.feature_name()
            metrics_path = model_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as fh:
                    self.metrics = json.load(fh)
            importance_path = model_path / "feature_importance.csv"
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path)

            prophet_path = model_path / "prophet_model.json"
            if prophet_path.exists() and Prophet is not None:
                with open(prophet_path, "r") as fh:
                    prophet_json = json.load(fh)
                self.temporal_model = Prophet.from_json(prophet_json)
                self.temporal_backend = "prophet"
            return True
        except Exception as exc:  # pragma: no cover - load failures are logged
            logger.error("Error loading models: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Weather helpers (retained from original implementation)
    # ------------------------------------------------------------------ #
    async def fetch_weather_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, float]]:
        try:
            import aiohttp

            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    payload = await response.json()

            daily = payload.get("daily", {})
            dates = daily.get("time", [])
            result: Dict[str, Dict[str, float]] = {}
            for idx, date in enumerate(dates):
                result[date] = {
                    "temp_max": daily.get("temperature_2m_max", [None])[idx],
                    "temp_min": daily.get("temperature_2m_min", [None])[idx],
                    "precip": daily.get("precipitation_sum", [None])[idx],
                    "evapotrans": daily.get("et0_fao_evapotranspiration", [None])[idx],
                }
            return result
        except Exception as exc:  # pragma: no cover - network failures logged upstream
            logger.error("Error fetching weather data: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _resolve_crop_name(self, crop_type: str) -> str:
        if not crop_type:
            return "Other"
        for key in CROP_BASELINE_YIELD.keys():
            if key.lower() == crop_type.lower():
                return key
        return "Other"


def run_coroutine(coro: asyncio.Future) -> Any:
    """Execute async methods from synchronous code."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
