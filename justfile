# This is a justfile

default:
  @just --list

run:
  poetry run algo-trade

install:
  poetry install

update:
  poetry update

test:
  poetry run python -m pytest tests -v

black:
  poetry run black algo_trade/data_engine
  poetry run black tests

help:
  @just --list
