import torch
from torch.autograd import Variable


def test_profiler():
    x = torch.randn((1, 1), requires_grad=True)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = x ** 2
        y.backward()
    print(prof)


def test_profiler_two():
    x = Variable(torch.randn(5,5), requires_grad=True).cuda()
    with torch.autograd.profiler.profile() as prof:
        y = x**2

        with torch.autograd.profiler.emit_nvtx():
            y = x**2
    print(prof)


def main():
    test_profiler()
#    test_profiler_two()


if __name__ == "__main__":
    main()

