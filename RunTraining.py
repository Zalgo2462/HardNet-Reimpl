from MinBatchNegativeMiner import MinBatchNegativeMiner

from HardNet import HardNet
from HardNetModule import HardNetModule


def main():
    hard_net = HardNet(MinBatchNegativeMiner(), HardNetModule())
    pass


if __name__ == "__main__":
    main()
