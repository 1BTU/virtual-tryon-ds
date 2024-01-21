
class BasicBlock():
    def __init__(self,layer_name:str,state_dict:list,module_name:list):
        self.layer_name = layer_name
        self.state_dict = [s for s in state_dict if len(state_dict) != 0]
        self.module_name = module_name
        self.num_layer = 0
        self.weight = 0
        self.type = None

class ResBlock_ifo(BasicBlock):
    def __init__(self,layer_name:str,state_dict:list,module_name:list):
        super().__init__(layer_name,state_dict,module_name)
        self.in_channel1 = state_dict[5].shape[0]
        self.out_channel1 = state_dict[5].shape[1]
        self.out_channel2 = state_dict[-1].shape[0]
        self.num_layer = 2
        self.out_weight1 = self.state_dict[5]
        self.out_weight2 = self.state_dict[-1]

class DownSample_ifo(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list, module_name: list):
        super().__init__(layer_name, state_dict, module_name)
        self.in_channel = state_dict[-1].shape[1]
        self.out_channel = state_dict[-1].shape[0]
        self.weight = state_dict[-1]
class MFEBlock_ifo(BasicBlock):
    def __init__(self, layer_name: str, state_dict: list, module_name: list):
        super().__init__(layer_name, state_dict, module_name)
        self.in_channel1 = state_dict[0].shape[1]
        self.out_channel1 = state_dict[0].shape[0]
        self.out_channel2 = state_dict[2].shape[0]
        self.out_channel3 = state_dict[4].shape[0]
        self.out_channel4 = state_dict[6].shape[0]

        self.weight1 = state_dict[0]
        self.weight2 = state_dict[2]
        self.weight3 = state_dict[4]
        self.weight4 = state_dict[6]
