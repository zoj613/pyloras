.PHONY: clean install test sdist wheels

clean:
	rm -Rf build/* dist/* ./pyloras.egg-info **/*__pycache__ __pycache__ .coverage*

dev:
	pip install -r requirements-dev.txt

sdist:
	python -m build --sdist

wheel:
	python -m build --wheel

test:
	pytest tests/ -vvv

test-cov: clean
	pytest tests/ -vv --cov-branch --cov=pyloras tests/ --cov-report=html
