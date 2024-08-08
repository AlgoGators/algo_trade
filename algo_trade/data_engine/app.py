"""
The Data Engine has three main responsibilities:
1. Data Ingestion
  - Read data from a source and confirm all required data is present
  - Transform data into a format that can be used by the Data Model
2. Data Refreshing
  - Upload the data to the remote database
3. Data Validation
"""

import threading

from flask import Flask, jsonify, request
from pipeline import Pipeline


def create_app():
    # Create the app and load the pipeline for requests
    app = Flask(__name__)

    p = Pipeline()
    # p.rebuild()
    p.transform()
    p.load()

    @app.route("/get_trend_tables", methods=["GET"])
    def get_trend_tables():
        trend_tables = p.get_trend_tables()
        trend_tables_json = {
            contract: table.to_json() for contract, table in trend_tables.items()
        }
        return jsonify(trend_tables_json)

    @app.route("/get_positions", methods=["GET"])
    def get_positions():
        signals = p.signals()
        return jsonify(signals)

    @app.route("/healthcheck", methods=["GET"])
    def healthcheck():
        return jsonify({"status": "ok"})

    return app
