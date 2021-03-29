import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../protobuf/")
from cost_model import CostModel

cost_model = CostModel()
cost_model.test_accuracy()
