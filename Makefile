.PHONY: test

test:
	venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
