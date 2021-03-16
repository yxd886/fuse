import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../protobuf/")
from cost_model import CostModel
base_cost_model = CostModel(profiler_path="base_profile",hlo_module_path="base_hlo.pb")
search_cost_model = CostModel(profiler_path="search_profile",hlo_module_path="search_hlo.pb")

print("Base cost model:")
base_cost_model.test_accuracy()

print("Search cost model:")
search_cost_model.test_accuracy()