import sys
from cost_model import CostModel

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../protobuf/")
cost_model = CostModel()
cost_model.test_accuracy()
