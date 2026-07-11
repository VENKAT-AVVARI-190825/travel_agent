"""
Weather forecasting tool using the Open-Meteo API.

Error-handling design (see interview Q8): instead of one broad ``except
Exception`` that treats every failure the same, this module:
  - distinguishes **retryable** faults (network timeouts, connection drops,
    429/5xx) from **terminal** ones (bad input, 4xx, location not found);
  - retries retryable faults with exponential backoff;
  - logs through the ``logging`` module (not ``print``) so failures are
    observable in a service;
  - returns a structured, fail-soft ``{"error", "error_type", "retryable"}``
    dict so the calling agent node can decide whether to retry or fall back.
"""
import time
import logging
import requests
from datetime import datetime, date

logger = logging.getLogger(__name__)


class WeatherTool:
    """Weather forecasting tool using Open-Meteo API"""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    # Transient HTTP statuses worth retrying; everything else is terminal.
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    MAX_RETRIES = 3
    BASE_BACKOFF = 0.5  # seconds; doubled each attempt

    # -- internal helpers ---------------------------------------------------

    def _http_get(self, url: str, params: dict, timeout: int = 10):
        """GET with retry + backoff on transient failures.

        Returns ``(json_data, None)`` on success or ``(None, error_dict)`` on
        terminal failure. Retries only faults that could plausibly succeed on a
        second attempt; returns immediately on client (4xx) errors.
        """
        last_reason = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Transient network fault -> back off and retry.
                last_reason = str(e)
                self._backoff(attempt, f"network error: {e}")
                continue

            if resp.ok:
                try:
                    return resp.json(), None
                except ValueError as e:
                    # Server returned non-JSON: retrying won't help -> terminal.
                    logger.error("Weather GET %s returned invalid JSON: %s", url, e)
                    return None, self._err("Weather service returned an invalid response",
                                           "bad_response", retryable=False)

            if resp.status_code in self.RETRYABLE_STATUS:
                last_reason = f"HTTP {resp.status_code}"
                self._backoff(attempt, f"retryable status {resp.status_code}")
                continue

            # Terminal client/other HTTP error -> do not retry.
            logger.error("Weather GET %s terminal status %s", url, resp.status_code)
            return None, self._err(f"Weather service error (status {resp.status_code})",
                                   "client_error", retryable=False)

        # Exhausted retries on a transient fault.
        logger.error("Weather GET %s failed after %d retries (%s)", url, self.MAX_RETRIES, last_reason)
        return None, self._err(f"Weather service unavailable after {self.MAX_RETRIES} retries",
                               "unavailable", retryable=True)

    def _backoff(self, attempt: int, reason: str) -> None:
        wait = self.BASE_BACKOFF * (2 ** (attempt - 1))
        logger.warning("Weather request attempt %d/%d failed (%s); retrying in %.1fs",
                       attempt, self.MAX_RETRIES, reason, wait)
        time.sleep(wait)

    @staticmethod
    def _err(message: str, error_type: str, retryable: bool) -> dict:
        """Build a structured, fail-soft error the agent can branch on."""
        return {"error": message, "error_type": error_type, "retryable": retryable}

    def _geocode(self, place: str):
        """Resolve a place name to coordinates. Returns (location, None) or (None, error)."""
        data, err = self._http_get(self.GEOCODING_URL, {"name": place.strip(), "count": 1})
        if err:
            return None, err
        results = data.get("results")
        if not results:
            return None, self._err(
                f"Location '{place}' not found - try a different spelling or nearby city",
                "not_found", retryable=False)
        info = results[0]
        return {
            "latitude": info["latitude"],
            "longitude": info["longitude"],
            "name": info["name"],
            "country": info.get("country", "Unknown"),
        }, None

    def _parse_forecast(self, weather_data: dict, location: dict,
                        extra_fields: bool = False, date_range: str = None):
        """Shape Open-Meteo daily data into our forecast contract.

        Parsing/shape bugs are caught narrowly (KeyError/TypeError/IndexError)
        and tagged 'parse_error' so they are distinguishable from network faults.
        """
        try:
            daily = weather_data.get("daily", {})
            if not daily.get("time"):
                return self._err("No weather data available for this location",
                                 "no_data", retryable=False)
            forecast = []
            for i, date_str in enumerate(daily["time"]):
                entry = {
                    "date": date_str,
                    "temp_max": daily["temperature_2m_max"][i],
                    "temp_min": daily["temperature_2m_min"][i],
                    "precipitation": daily["precipitation_sum"][i],
                }
                if extra_fields:
                    entry["weather_code"] = daily["weather_code"][i]
                    entry["wind_speed_max"] = daily["wind_speed_10m_max"][i]
                forecast.append(entry)
            result = {"location": location, "forecast": forecast}
            if date_range:
                result["date_range"] = date_range
            return result
        except (KeyError, TypeError, IndexError) as e:
            logger.exception("Weather response parsing failed")
            return self._err(f"Could not parse weather response: {e}",
                             "parse_error", retryable=False)

    # -- public API (signatures & return contract unchanged) ----------------

    def get_weather(self, place: str, days: int = 3):
        """
        Get weather forecast for a number of days.

        Returns:
            {"location": {...}, "forecast": [...]} on success, or
            {"error": str, "error_type": str, "retryable": bool} on failure.
        """
        if not place or not place.strip():
            return self._err("Location name cannot be empty", "invalid_input", retryable=False)
        if days < 1 or days > 16:
            return self._err("Forecast days must be between 1 and 16", "invalid_input", retryable=False)

        location, err = self._geocode(place)
        if err:
            return err

        data, err = self._http_get(self.WEATHER_URL, {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto",
            "forecast_days": days,
        })
        if err:
            return err
        return self._parse_forecast(data, location)

    def get_weather_range(self, place: str, start_date: str, end_date: str):
        """
        Get weather forecast for a specific date range (YYYY-MM-DD).

        Returns:
            {"location": {...}, "forecast": [...], "date_range": str} on success,
            or {"error": str, "error_type": str, "retryable": bool} on failure.
        """
        if not place or not place.strip():
            return self._err("Location name cannot be empty", "invalid_input", retryable=False)
        if not start_date or not end_date:
            return self._err("Both start_date and end_date are required", "invalid_input", retryable=False)

        # Validate date format/logic -- terminal input errors, never retried.
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return self._err("Invalid date format - use YYYY-MM-DD (e.g., 2025-06-15)",
                             "invalid_input", retryable=False)
        if start_dt > end_dt:
            return self._err("Start date must be before or equal to end date",
                             "invalid_input", retryable=False)
        if start_dt < date.today():
            return self._err("Start date cannot be in the past", "invalid_input", retryable=False)
        if (end_dt - start_dt).days + 1 > 16:
            return self._err("Date range too long - maximum 16 days forecast available",
                             "invalid_input", retryable=False)

        location, err = self._geocode(place)
        if err:
            return err

        data, err = self._http_get(self.WEATHER_URL, {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                      "weather_code", "wind_speed_10m_max"],
            "timezone": "auto",
            "start_date": start_date,
            "end_date": end_date,
        })
        if err:
            return err
        return self._parse_forecast(data, location, extra_fields=True,
                                    date_range=f"{start_date} to {end_date}")
