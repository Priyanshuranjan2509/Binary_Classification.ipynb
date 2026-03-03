import os
import sys
import tempfile

# ensure `src` package is importable during tests
# add project root or src directory to sys.path so imports succeed
# using cwd instead of __file__ path, which can behave oddly inside some test runners
root = os.getcwd()
# add project root so that `src` package can be imported normally
p = root
print("in test, inserting project root", p)
sys.path.insert(0, p)
print("sys.path now", sys.path[:5])

from src.binary_classification import model
print("import model succeeded", model)


def test_train_and_load():
    # train on small dataset and ensure model file is created and loadable
    with tempfile.TemporaryDirectory() as tmpdir:
        output = os.path.join(tmpdir, "test_model.joblib")
        clf, acc = model.train(output_path=output, n_samples=100, n_features=5)
        assert os.path.exists(output)
        loaded = model.load_model(output)
        assert loaded.predict([[0]*5]) is not None  # simple smoke test
        assert 0 <= acc <= 1
