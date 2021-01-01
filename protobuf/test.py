import hlo_pb2
files = ["0_0_before_optimizations.hlo","0_0_after_optimizations.hlo","3_0_before_optimizations.hlo","3_0_after_optimizations.hlo","3_60_before_optimizations.hlo","3_60_after_optimizations.hlo"]
for file in files:
    with open(file+".pb","rb") as f:
        hlo_module = hlo_pb2.HloProto()
        hlo_module.ParseFromString(f.read())
    with open(file+".pbtxt","w") as f:
        f.write(str(hlo_module))