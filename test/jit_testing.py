import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)

        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = self.linear(output)
        return output

scripted_module = torch.jit.script(MyModule(10,20))
