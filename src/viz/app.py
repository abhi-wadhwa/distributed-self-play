"""
Simple web dashboard for monitoring distributed self-play training.

Displays:
  - Training loss over time
  - Games per second
  - ELO progression across generations
  - Replay buffer utilization
  - Current generation and training step
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from flask import Flask, Response, jsonify, render_template_string

from src.communication import MockRedisInterface, RedisInterface

logger = logging.getLogger(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Play Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117;
            color: #e1e4e8;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #58a6ff;
            font-size: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        .metric {
            font-size: 36px;
            font-weight: bold;
            color: #58a6ff;
        }
        .metric-small {
            font-size: 14px;
            color: #8b949e;
            margin-top: 4px;
        }
        .chart-placeholder {
            width: 100%;
            height: 120px;
            background: #0d1117;
            border-radius: 4px;
            display: flex;
            align-items: flex-end;
            padding: 4px;
            gap: 2px;
            overflow: hidden;
        }
        .bar {
            flex: 1;
            background: #238636;
            border-radius: 2px 2px 0 0;
            min-width: 3px;
            transition: height 0.3s ease;
        }
        .elo-entry {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #21262d;
            font-size: 14px;
        }
        .elo-entry:last-child { border-bottom: none; }
        .elo-rating { color: #f0883e; font-weight: bold; }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background: #238636; }
        .status-idle { background: #f0883e; }
        #refresh-info {
            text-align: center;
            color: #484f58;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Distributed Self-Play Training</h1>
    <div class="grid">
        <div class="card">
            <h2>Generation</h2>
            <div class="metric" id="generation">--</div>
            <div class="metric-small" id="training-step">Step: --</div>
        </div>
        <div class="card">
            <h2>Training Loss</h2>
            <div class="metric" id="total-loss">--</div>
            <div class="metric-small">
                Policy: <span id="policy-loss">--</span> |
                Value: <span id="value-loss">--</span>
            </div>
        </div>
        <div class="card">
            <h2>Buffer Size</h2>
            <div class="metric" id="buffer-size">--</div>
            <div class="metric-small" id="buffer-util">Utilization: --%</div>
        </div>
        <div class="card">
            <h2>Games / Second</h2>
            <div class="metric" id="games-per-sec">--</div>
            <div class="metric-small" id="total-experiences">Total experiences: --</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <h2>Loss History</h2>
            <div class="chart-placeholder" id="loss-chart"></div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <h2>ELO Leaderboard</h2>
            <div id="elo-board">
                <div class="elo-entry">
                    <span>No data yet</span>
                    <span class="elo-rating">--</span>
                </div>
            </div>
        </div>
    </div>
    <div id="refresh-info">Auto-refreshes every 3 seconds</div>

    <script>
        async function fetchData() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();

                // Update metrics
                document.getElementById('generation').textContent = data.generation || '--';
                document.getElementById('training-step').textContent =
                    'Step: ' + (data.training_step || '--');

                if (data.latest_loss) {
                    document.getElementById('total-loss').textContent =
                        data.latest_loss.total_loss.toFixed(4);
                    document.getElementById('policy-loss').textContent =
                        data.latest_loss.policy_loss.toFixed(4);
                    document.getElementById('value-loss').textContent =
                        data.latest_loss.value_loss.toFixed(4);
                }

                document.getElementById('buffer-size').textContent =
                    (data.buffer_size || 0).toLocaleString();
                document.getElementById('buffer-util').textContent =
                    'Utilization: ' + ((data.buffer_utilization || 0) * 100).toFixed(1) + '%';

                document.getElementById('games-per-sec').textContent =
                    (data.games_per_sec || 0).toFixed(1);
                document.getElementById('total-experiences').textContent =
                    'Total experiences: ' + (data.total_experiences || 0).toLocaleString();

                // Loss chart
                if (data.loss_history && data.loss_history.length > 0) {
                    const chart = document.getElementById('loss-chart');
                    const maxLoss = Math.max(...data.loss_history.map(l => l.total_loss));
                    chart.innerHTML = data.loss_history.slice(-60).map(l => {
                        const h = Math.max(4, (l.total_loss / maxLoss) * 100);
                        return '<div class="bar" style="height:' + h + '%"></div>';
                    }).join('');
                }

                // ELO leaderboard
                if (data.elo_history && Object.keys(data.elo_history).length > 0) {
                    const board = document.getElementById('elo-board');
                    const sorted = Object.entries(data.elo_history)
                        .sort((a, b) => b[1] - a[1]);
                    board.innerHTML = sorted.map(([gen, elo]) =>
                        '<div class="elo-entry">' +
                        '<span>Generation ' + gen + '</span>' +
                        '<span class="elo-rating">' + Math.round(elo) + '</span>' +
                        '</div>'
                    ).join('');
                }

            } catch (e) {
                console.error('Failed to fetch data:', e);
            }
        }

        fetchData();
        setInterval(fetchData, 3000);
    </script>
</body>
</html>
"""


def create_app(
    comm: Union[RedisInterface, MockRedisInterface],
) -> Flask:
    """
    Create the Flask dashboard application.

    Args:
        comm: Redis communication interface for fetching metrics.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/status")
    def api_status() -> Response:
        """Return current training status as JSON."""
        metrics = comm.get_metrics(count=200)

        # Extract latest training metrics (not evaluation metrics)
        training_metrics = [
            m for m in metrics if m.get("type") != "evaluation"
        ]
        latest = training_metrics[-1] if training_metrics else {}

        # Loss history
        loss_history = [
            {
                "total_loss": m.get("total_loss", 0),
                "policy_loss": m.get("policy_loss", 0),
                "value_loss": m.get("value_loss", 0),
            }
            for m in training_metrics
            if "total_loss" in m
        ]

        # Compute games/sec from recent metrics
        games_per_sec = 0.0
        if len(training_metrics) >= 2:
            recent = training_metrics[-10:]
            if len(recent) >= 2:
                time_span = recent[-1].get("timestamp", 0) - recent[0].get("timestamp", 0)
                if time_span > 0:
                    exp_span = sum(
                        m.get("experiences_pulled", 0) for m in recent
                    )
                    # Rough estimate: average ~50 experiences per game
                    games_per_sec = (exp_span / 50.0) / time_span

        # ELO history
        elo_history = comm.get_elo_history()

        status = {
            "generation": latest.get("generation", 0),
            "training_step": latest.get("training_step", 0),
            "buffer_size": latest.get("buffer_size", 0),
            "buffer_utilization": latest.get("buffer_size", 0) / 500_000,
            "total_experiences": sum(
                m.get("experiences_pulled", 0) for m in training_metrics
            ),
            "games_per_sec": games_per_sec,
            "latest_loss": {
                "total_loss": latest.get("total_loss", 0),
                "policy_loss": latest.get("policy_loss", 0),
                "value_loss": latest.get("value_loss", 0),
            } if "total_loss" in latest else None,
            "loss_history": loss_history[-60:],
            "elo_history": {str(k): v for k, v in elo_history.items()},
        }
        return jsonify(status)

    @app.route("/api/health")
    def health() -> Response:
        """Health check endpoint."""
        connected = comm.ping()
        return jsonify({"status": "ok" if connected else "degraded", "redis": connected})

    return app


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 5000,
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> None:
    """Start the monitoring dashboard server."""
    comm = RedisInterface(host=redis_host, port=redis_port)
    app = create_app(comm)
    logger.info(f"Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
